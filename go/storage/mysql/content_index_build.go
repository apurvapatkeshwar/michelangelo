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
// resource.revisioned_in / index annotations off every CRD type registered in the
// scheme. A base type's revisioned_in lists the wrapper kinds it is snapshotted
// into (e.g. "revision", "draft"); each kind resolves to a wrapper CRD by
// convention (same group/version, Kind = TitleCase(kind), keyed on <kind>_uid) and
// the wrapped resource always lives at spec.content. Returns nil when no content
// sidecars apply, so the storage stays in plain mode.
//
// This is the runtime equivalent of what protoc-gen-sql does at codegen time:
// the same revisioned_in / index annotations that generate the "*_unmarshalled"
// tables also tell the storage how to populate (writes) and filter (reads) them.
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
		if res := resourceOf(opts); res != nil && len(res.GetRevisionedIn()) > 0 {
			var fields []indexField
			for _, ix := range indexesOf(opts) {
				fields = append(fields, indexField{path: ix.GetPath(), key: ix.GetKey()})
			}
			bases = append(bases, contentBaseInfo{
				gvk:          gvk,
				revisionedIn: res.GetRevisionedIn(),
				indexFields:  fields,
			})
		}
	}

	specs := crossJoinContentSpecs(bases)
	if len(specs) == 0 {
		return nil
	}
	return BuildContentIndex(specs)
}

type contentBaseInfo struct {
	gvk          schema.GroupVersionKind
	revisionedIn []string
	indexFields  []indexField
}

type indexField struct {
	path string
	key  string
}

// crossJoinContentSpecs emits one ContentIndexFieldSpec per (base, wrapper-kind)
// pair listed in each base type's revisioned_in. The wrapper kind resolves to a
// wrapper CRD by convention; the table name, uid column, and filterable field
// paths are derived exactly as the codegen does:
// table = <baseKind>_<wrapperKind>_unmarshalled, uid col = <wrapperKind>_uid,
// field path = spec.content.<base index path>, column = base index key. Pure so it
// is unit-testable without proto descriptors.
func crossJoinContentSpecs(bases []contentBaseInfo) []ContentIndexFieldSpec {
	var specs []ContentIndexFieldSpec
	for _, b := range bases {
		for _, kind := range b.revisionedIn {
			fields := make([]ContentIndexField, 0, len(b.indexFields))
			for _, f := range b.indexFields {
				fields = append(fields, ContentIndexField{
					Path:   contentWrapperContentPath + "." + f.path,
					Column: f.key,
				})
			}
			specs = append(specs, ContentIndexFieldSpec{
				WrapperGVK: schema.GroupVersionKind{
					Group:   b.gvk.Group,
					Version: b.gvk.Version,
					Kind:    titleKind(kind),
				},
				ContentPath: contentWrapperContentPath,
				BaseKind:    b.gvk.Kind,
				Table:       utils.ToSnakeCase(b.gvk.Kind) + "_" + kind + "_unmarshalled",
				UIDCol:      kind + "_uid",
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

func indexesOf(opts *gogodesc.MessageOptions) []*apipb.IndexDescriptor {
	v, err := gogoproto.GetExtension(opts, apipb.E_Index)
	if err != nil {
		return nil
	}
	idx, _ := v.([]*apipb.IndexDescriptor)
	return idx
}
