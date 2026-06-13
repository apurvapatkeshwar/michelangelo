package mysql

import (
	"reflect"

	gogoproto "github.com/gogo/protobuf/proto"
	gogodesc "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/michelangelo-ai/michelangelo/go/api/utils"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// BuildContentIndexFromScheme constructs the ContentIndex by reading the
// content_wrapper / resource.revisioned_in / index annotations off every CRD
// type registered in the scheme, then cross-joining wrappers with the revisioned
// base types that opted them in. Returns nil when no content sidecars apply, so
// the storage stays in plain mode.
//
// This is the runtime equivalent of what protoc-gen-sql does at codegen time:
// the same annotations that generate the "*_unmarshalled" tables also tell the
// storage how to populate (writes) and filter (reads) them.
func BuildContentIndexFromScheme(scheme *runtime.Scheme) *ContentIndex {
	if scheme == nil {
		return nil
	}

	var wrappers []contentWrapperInfo
	var bases []contentBaseInfo
	for gvk, rt := range scheme.AllKnownTypes() {
		opts := messageOptions(reflect.New(rt).Interface())
		if opts == nil {
			continue
		}
		if cw := contentWrapperOf(opts); cw != nil && cw.GetContentPath() != "" {
			wrappers = append(wrappers, contentWrapperInfo{
				gvk:         gvk,
				contentPath: cw.GetContentPath(),
				kind:        cw.GetKind(),
			})
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

	specs := crossJoinContentSpecs(wrappers, bases)
	if len(specs) == 0 {
		return nil
	}
	return BuildContentIndex(specs)
}

type contentWrapperInfo struct {
	gvk         schema.GroupVersionKind
	contentPath string
	kind        string
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

// crossJoinContentSpecs emits one ContentIndexFieldSpec per (wrapper, base) pair
// where the base opted the wrapper's kind into its revisioned_in. The table name,
// uid column, and filterable field paths are derived exactly as the codegen does:
// table = <baseKind>_<wrapperKind>_unmarshalled, uid col = <wrapperKind>_uid,
// field path = <content_path>.<base index path>, column = base index key. Pure so
// it is unit-testable without proto descriptors.
func crossJoinContentSpecs(wrappers []contentWrapperInfo, bases []contentBaseInfo) []ContentIndexFieldSpec {
	var specs []ContentIndexFieldSpec
	for _, w := range wrappers {
		for _, b := range bases {
			if !containsString(b.revisionedIn, w.kind) {
				continue
			}
			fields := make([]ContentIndexField, 0, len(b.indexFields))
			for _, f := range b.indexFields {
				fields = append(fields, ContentIndexField{
					Path:   w.contentPath + "." + f.path,
					Column: f.key,
				})
			}
			specs = append(specs, ContentIndexFieldSpec{
				WrapperGVK:  w.gvk,
				ContentPath: w.contentPath,
				BaseKind:    b.gvk.Kind,
				Table:       utils.ToSnakeCase(b.gvk.Kind) + "_" + utils.ToSnakeCase(w.gvk.Kind) + "_unmarshalled",
				UIDCol:      utils.ToSnakeCase(w.gvk.Kind) + "_uid",
				Fields:      fields,
			})
		}
	}
	return specs
}

func containsString(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
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

func contentWrapperOf(opts *gogodesc.MessageOptions) *apipb.ContentWrapperDescriptor {
	v, err := gogoproto.GetExtension(opts, apipb.E_ContentWrapper)
	if err != nil {
		return nil
	}
	cw, _ := v.(*apipb.ContentWrapperDescriptor)
	return cw
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
