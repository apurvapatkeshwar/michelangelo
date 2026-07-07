package mysql

import (
	"reflect"

	"github.com/michelangelo-ai/michelangelo/go/storage"
	"k8s.io/apimachinery/pkg/runtime"
)

// BuildContentIndexFromScheme constructs the ContentIndex by asking every CRD
// type registered in the scheme for its own content-index specs. A revisioned
// base type implements storage.ContentIndexDescribable, generated directly from
// its resource.revisioned_in annotation — the same parse that produces the
// sidecar DDL and GetContentIndexedKeyValuePairs, with the wrapper GVK, table,
// and uid column already resolved and validated at codegen time. This function
// never re-parses revisioned_in itself: it only discovers which types are
// registered in this process and collects what they report, so the DDL, the
// extractor, and this runtime routing can never drift apart. Returns nil when
// no type reports any specs, so the storage stays in plain mode.
func BuildContentIndexFromScheme(scheme *runtime.Scheme) *ContentIndex {
	if scheme == nil {
		return nil
	}

	var specs []ContentIndexFieldSpec
	for _, rt := range scheme.AllKnownTypes() {
		describable, ok := reflect.New(rt).Interface().(storage.ContentIndexDescribable)
		if !ok {
			continue
		}
		specs = append(specs, describable.ContentIndexFieldSpecs()...)
	}

	if len(specs) == 0 {
		return nil
	}
	return BuildContentIndex(specs)
}
