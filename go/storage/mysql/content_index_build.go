package mysql

import (
	"reflect"
	"strings"

	gogoproto "github.com/gogo/protobuf/proto"
	gogodesc "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/michelangelo-ai/michelangelo/go/api/utils"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// contentWrapperContentPath is where every content-wrapper CRD (Revision, Draft)
// stores its wrapped base resource as a google.protobuf.Any. This is a fixed
// convention shared with protoc-gen-sql, so the wrapper CRD does not annotate it.
const contentWrapperContentPath = "spec.content"

// BuildContentIndexFromScheme constructs the ContentIndex by reading the
// resource.revisioned_in annotations off every CRD type registered in the
// scheme. A base type's revisioned_in lists the wrapper kinds it is snapshotted
// into (e.g. "revision", "draft"); each entry carries its own content_index list
// declaring the content fields to project into that wrapper's sidecar. Each kind
// resolves to a wrapper CRD by convention (same group/version, Kind =
// TitleCase(kind), keyed on <kind>_uid) and the wrapped resource always lives at
// spec.content. Returns nil when no content sidecars apply, so the storage stays
// in plain mode.
//
// This is the runtime equivalent of what protoc-gen-sql does at codegen time:
// the same revisioned_in[].content_index annotations that generate the
// "*_unmarshalled" tables also tell the storage how to populate (writes) and
// filter (reads) them. The base type's own index annotations are NOT consulted
// here — they drive only the base table.
func BuildContentIndexFromScheme(scheme *runtime.Scheme) *ContentIndex {
	if scheme == nil {
		return nil
	}

	var bases []contentBaseInfo
	for gvk, rt := range scheme.AllKnownTypes() {
		opts := messageOptions(reflect.New(rt).Interface())
		if opts == nil {
			continue
		}
		res := resourceOf(opts)
		if res == nil || len(res.GetRevisionedIn()) == 0 {
			continue
		}
		var wrappers []wrapperContentInfo
		for _, ri := range res.GetRevisionedIn() {
			var fields []indexField
			for _, ci := range ri.GetContentIndex() {
				fields = append(fields, indexField{path: ci.GetPath(), key: ci.GetKey()})
			}
			wrappers = append(wrappers, wrapperContentInfo{kind: ri.GetKind(), fields: fields})
		}
		bases = append(bases, contentBaseInfo{gvk: gvk, wrappers: wrappers})
	}

	specs := crossJoinContentSpecs(bases)
	if len(specs) == 0 {
		return nil
	}
	return BuildContentIndex(specs)
}

type contentBaseInfo struct {
	gvk      schema.GroupVersionKind
	wrappers []wrapperContentInfo
}

// wrapperContentInfo is one revisioned_in entry: the wrapper kind plus the
// content fields (paths relative to the decoded content) projected into its
// sidecar table.
type wrapperContentInfo struct {
	kind   string
	fields []indexField
}

type indexField struct {
	path string
	key  string
}

// crossJoinContentSpecs emits one ContentIndexFieldSpec per (base, wrapper)
// entry. Each wrapper carries its own content_index list, so wrappers can have
// different column subsets. The wrapper kind resolves to a wrapper CRD by
// convention; the table name, uid column, and filterable field paths are derived
// exactly as the codegen does: table = <baseKind>_<wrapperKind>_unmarshalled,
// uid col = <wrapperKind>_uid, field path = spec.content.<content path>, column =
// content key. Pure so it is unit-testable without proto descriptors.
func crossJoinContentSpecs(bases []contentBaseInfo) []ContentIndexFieldSpec {
	var specs []ContentIndexFieldSpec
	for _, b := range bases {
		for _, w := range b.wrappers {
			fields := make([]ContentIndexField, 0, len(w.fields))
			for _, f := range w.fields {
				fields = append(fields, ContentIndexField{
					Path:   contentWrapperContentPath + "." + f.path,
					Column: f.key,
				})
			}
			specs = append(specs, ContentIndexFieldSpec{
				WrapperGVK: schema.GroupVersionKind{
					Group:   b.gvk.Group,
					Version: b.gvk.Version,
					Kind:    titleKind(w.kind),
				},
				WrapperKind: w.kind,
				ContentPath: contentWrapperContentPath,
				BaseKind:    b.gvk.Kind,
				Table:       utils.ToSnakeCase(b.gvk.Kind) + "_" + w.kind + "_unmarshalled",
				UIDCol:      w.kind + "_uid",
				Fields:      fields,
			})
		}
	}
	return specs
}

// titleKind upper-cases the first letter of a wrapper kind so the kind string
// "revision" resolves to the wrapper CRD Kind "Revision".
func titleKind(kind string) string {
	if kind == "" {
		return ""
	}
	return strings.ToUpper(kind[:1]) + kind[1:]
}

// messageOptions returns the gogo MessageOptions for a CRD object, or nil if the
// object isn't a gogo proto message (e.g. a non-CRD type registered in the scheme).
func messageOptions(obj interface{}) *gogodesc.MessageOptions {
	msg, ok := obj.(gogodesc.Message)
	if !ok {
		return nil
	}
	_, md := gogodesc.ForMessage(msg)
	if md == nil {
		return nil
	}
	return md.GetOptions()
}

func resourceOf(opts *gogodesc.MessageOptions) *apipb.ResourceDescriptor {
	v, err := gogoproto.GetExtension(opts, apipb.E_Resource)
	if err != nil {
		return nil
	}
	res, _ := v.(*apipb.ResourceDescriptor)
	return res
}
