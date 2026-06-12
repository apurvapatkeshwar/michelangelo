package mysql

import (
	"context"
	"database/sql"
	"fmt"
	"reflect"
	"strings"

	proto "github.com/gogo/protobuf/proto"
	gogotypes "github.com/gogo/protobuf/types"
	"github.com/michelangelo-ai/michelangelo/go/storage"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ContentIndexFieldSpec is the single source of truth describing one content
// sidecar table for a (wrapper, base type) pair. It carries everything both the
// read path (path -> column lookups) and the write path (which table to populate)
// need. In production this is produced from the content_index proto annotations
// (codegen); it is plain data so it can also be hand-built or built at runtime.
type ContentIndexFieldSpec struct {
	// WrapperGVK identifies the wrapper CRD (e.g. Revision, Draft).
	WrapperGVK schema.GroupVersionKind
	// ContentPath is the path to the wrapper's google.protobuf.Any content field,
	// e.g. "spec.content".
	ContentPath string
	// BaseKind is the wrapped base type's kind, e.g. "Pipeline".
	BaseKind string
	// Table is the sidecar table, e.g. "pipeline_revision_unmarshalled".
	Table string
	// UIDCol is the sidecar's FK column back to the wrapper uid, e.g. "revision_uid".
	UIDCol string
	// Fields maps each filterable content path to its sidecar column. Used by the
	// read path; the write path derives values from the base type's own extractor.
	Fields []ContentIndexField
}

// ContentIndexField is one filterable field: the full criterion path the FE
// sends (e.g. "spec.content.spec.type") and the sidecar column it maps to.
type ContentIndexField struct {
	Path   string
	Column string
}

// ContentIndex holds the derived read and write structures consumed by the
// storage layer. Build it with BuildContentIndex.
type ContentIndex struct {
	// ReadMaps: wrapper GVK -> criterion path -> candidate sidecar entries.
	ReadMaps map[schema.GroupVersionKind]map[string][]contentIndexEntry
	// WriteSpecs: wrapper GVK -> how to populate its sidecar tables.
	WriteSpecs map[schema.GroupVersionKind]contentIndexWriteSpec
}

// contentIndexWriteSpec tells the write path, for one wrapper, where its content
// blob is and which sidecar table each wrapped base kind populates.
type contentIndexWriteSpec struct {
	contentPath string
	targets     map[string]contentIndexTarget // base kind -> target table
}

type contentIndexTarget struct {
	table  string
	uidCol string
}

// BuildContentIndex derives the read maps and write specs from the field specs.
// Both halves come from the same input, so the path-level filtering and the
// table-level population can never disagree.
func BuildContentIndex(specs []ContentIndexFieldSpec) *ContentIndex {
	ci := &ContentIndex{
		ReadMaps:   map[schema.GroupVersionKind]map[string][]contentIndexEntry{},
		WriteSpecs: map[schema.GroupVersionKind]contentIndexWriteSpec{},
	}
	for _, s := range specs {
		readMap := ci.ReadMaps[s.WrapperGVK]
		if readMap == nil {
			readMap = map[string][]contentIndexEntry{}
			ci.ReadMaps[s.WrapperGVK] = readMap
		}
		for _, f := range s.Fields {
			readMap[f.Path] = append(readMap[f.Path], contentIndexEntry{
				BaseType: s.BaseKind,
				Table:    s.Table,
				Column:   f.Column,
				UIDCol:   s.UIDCol,
			})
		}

		ws, ok := ci.WriteSpecs[s.WrapperGVK]
		if !ok {
			ws = contentIndexWriteSpec{contentPath: s.ContentPath, targets: map[string]contentIndexTarget{}}
		}
		ws.targets[s.BaseKind] = contentIndexTarget{table: s.Table, uidCol: s.UIDCol}
		ci.WriteSpecs[s.WrapperGVK] = ws
	}
	return ci
}

// ContentIndexRow is one sidecar table row to upsert.
type ContentIndexRow struct {
	Table   string
	UIDCol  string
	UID     string
	Columns []contentIndexColumn
}

type contentIndexColumn struct {
	Name  string
	Value interface{}
}

// contentIndexUpsertSQL builds the idempotent upsert for one sidecar row,
// mirroring fullUpsert's INSERT ... ON DUPLICATE KEY UPDATE (the sidecar has one
// row per wrapper uid, so this overwrites cleanly on update). Pure (no DB) so it
// is unit-testable.
func contentIndexUpsertSQL(row ContentIndexRow) (string, []interface{}) {
	cols := make([]string, 0, len(row.Columns)+1)
	placeholders := make([]string, 0, len(row.Columns)+1)
	args := make([]interface{}, 0, len(row.Columns)+1)
	updates := make([]string, 0, len(row.Columns))

	cols = append(cols, "`"+row.UIDCol+"`")
	placeholders = append(placeholders, "?")
	args = append(args, row.UID)

	for _, c := range row.Columns {
		cols = append(cols, "`"+c.Name+"`")
		placeholders = append(placeholders, "?")
		args = append(args, c.Value)
		updates = append(updates, "`"+c.Name+"` = VALUES(`"+c.Name+"`)")
	}

	query := "INSERT INTO `" + row.Table + "` (" + strings.Join(cols, ", ") + ") VALUES (" +
		strings.Join(placeholders, ", ") + ")"
	if len(updates) > 0 {
		query += " ON DUPLICATE KEY UPDATE " + strings.Join(updates, ", ")
	}
	return query, args
}

// upsertContentIndex writes the sidecar rows for an object within the caller's
// transaction (so they commit atomically with the main row, like labels).
func (m *mysqlMetadataStorage) upsertContentIndex(ctx context.Context, tx *sql.Tx, rows []ContentIndexRow) error {
	for _, row := range rows {
		query, args := contentIndexUpsertSQL(row)
		if _, err := tx.ExecContext(ctx, query, args...); err != nil {
			return status.Errorf(codes.Internal, "failed to upsert content index row in %s: %v", row.Table, err)
		}
	}
	return nil
}

// indexedPairsGetter is implemented by every generated CRD type; we reuse the
// base type's own indexed-field extractor to populate the sidecar columns,
// rather than generating a second extractor.
type indexedPairsGetter interface {
	GetIndexedKeyValuePairs() []storage.IndexedField
}

// contentIndexRows derives the sidecar rows for a wrapper object: it locates the
// wrapper's content blob, decodes it to the concrete base type, and reuses that
// base type's GetIndexedKeyValuePairs to get the column values. Returns nil (no
// error) when the object isn't a configured wrapper, has no content, holds a
// base kind the wrapper didn't opt into, or whose content can't be decoded —
// none of those are write failures.
func (m *mysqlMetadataStorage) contentIndexRows(object runtime.Object) ([]ContentIndexRow, error) {
	if m.contentIndexWriteSpecs == nil {
		return nil, nil
	}
	gvk := m.gvkForObject(object)
	spec, ok := m.contentIndexWriteSpecs[gvk]
	if !ok {
		return nil, nil
	}

	contentAny, err := anyAtPath(object, spec.contentPath)
	if err != nil || contentAny == nil {
		return nil, nil // no content to index
	}
	baseMsg, baseName, err := decodeContentAny(contentAny)
	if err != nil {
		return nil, nil // undecodable content (e.g. a kind we don't model) — skip
	}
	baseKind := baseName
	if i := strings.LastIndex(baseKind, "."); i >= 0 {
		baseKind = baseKind[i+1:]
	}
	target, ok := spec.targets[baseKind]
	if !ok {
		return nil, nil // this wrapper didn't opt this base kind in
	}

	getter, ok := baseMsg.(indexedPairsGetter)
	if !ok {
		return nil, nil
	}
	metaObj, err := getObjectMeta(object)
	if err != nil {
		return nil, err
	}

	pairs := getter.GetIndexedKeyValuePairs()
	cols := make([]contentIndexColumn, 0, len(pairs))
	for _, p := range pairs {
		cols = append(cols, contentIndexColumn{Name: p.Key, Value: p.Value})
	}
	return []ContentIndexRow{{
		Table:   target.table,
		UIDCol:  target.uidCol,
		UID:     string(metaObj.GetUID()),
		Columns: cols,
	}}, nil
}

// gvkForObject resolves the object's GVK, falling back to the scheme when the
// object carries an empty TypeMeta (controller-runtime strips it).
func (m *mysqlMetadataStorage) gvkForObject(object runtime.Object) schema.GroupVersionKind {
	gvk := object.GetObjectKind().GroupVersionKind()
	if !gvk.Empty() {
		return gvk
	}
	if m.scheme != nil {
		if kinds, _, err := m.scheme.ObjectKinds(object); err == nil && len(kinds) > 0 {
			return kinds[0]
		}
	}
	return gvk
}

// anyAtPath walks a dotted proto path (e.g. "spec.content") over the object's Go
// struct via reflection and returns the *Any it resolves to, or nil if any link
// is nil. Errors only when the path is structurally invalid for the type.
func anyAtPath(object runtime.Object, path string) (*gogotypes.Any, error) {
	v := reflect.ValueOf(object)
	for _, seg := range strings.Split(path, ".") {
		if v.Kind() == reflect.Ptr {
			if v.IsNil() {
				return nil, nil
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return nil, fmt.Errorf("content path %q: %q is not a struct field", path, seg)
		}
		v = v.FieldByName(pascalCase(seg))
		if !v.IsValid() {
			return nil, fmt.Errorf("content path %q: no field for segment %q", path, seg)
		}
	}
	if v.Kind() == reflect.Ptr && v.IsNil() {
		return nil, nil
	}
	any, ok := v.Interface().(*gogotypes.Any)
	if !ok {
		return nil, fmt.Errorf("content path %q does not resolve to *Any (got %s)", path, v.Type())
	}
	return any, nil
}

// decodeContentAny resolves the concrete gogo proto message named by the Any's
// type_url and unmarshals into it. Returns the message and its full proto name.
func decodeContentAny(any *gogotypes.Any) (proto.Message, string, error) {
	name := any.TypeUrl
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	mt := proto.MessageType(name)
	if mt == nil || mt.Kind() != reflect.Ptr {
		return nil, "", fmt.Errorf("unknown content type %q", name)
	}
	msg, ok := reflect.New(mt.Elem()).Interface().(proto.Message)
	if !ok {
		return nil, "", fmt.Errorf("content type %q is not a proto.Message", name)
	}
	if err := gogotypes.UnmarshalAny(any, msg); err != nil {
		return nil, "", err
	}
	return msg, name, nil
}

// pascalCase converts a snake_case proto field name to its generated Go field
// name (e.g. "base_resource" -> "BaseResource", "content" -> "Content").
func pascalCase(s string) string {
	parts := strings.Split(s, "_")
	for i, p := range parts {
		if p != "" {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, "")
}
