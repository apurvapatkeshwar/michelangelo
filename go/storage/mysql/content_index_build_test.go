package mysql

import (
	"testing"

	"github.com/michelangelo-ai/michelangelo/go/storage"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// fakeRevisionedPipeline is a minimal storage.ContentIndexDescribable used to
// prove BuildContentIndexFromScheme's aggregation without depending on real
// generated proto code. Its specs are hand-built the way codegen would produce
// them: per-wrapper, with an asymmetric field subset (revision indexes state,
// draft does not) to prove wrappers keep independent column sets.
type fakeRevisionedPipeline struct {
	metav1.TypeMeta
}

func (f *fakeRevisionedPipeline) DeepCopyObject() runtime.Object {
	cp := *f
	return &cp
}

func (f *fakeRevisionedPipeline) ContentIndexFieldSpecs() []storage.ContentIndexFieldSpec {
	return []storage.ContentIndexFieldSpec{
		{
			WrapperGVK:  schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Revision"},
			WrapperKind: "revision",
			ContentPath: "spec.content",
			BaseKind:    "Pipeline",
			Table:       "pipeline_revision_unmarshalled",
			UIDCol:      "revision_uid",
			Fields: []storage.ContentIndexField{
				{Path: "spec.content.metadata.name", Column: "name"},
				{Path: "spec.content.spec.type", Column: "type"},
				{Path: "spec.content.status.state", Column: "state"},
			},
		},
		{
			WrapperGVK:  schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Draft"},
			WrapperKind: "draft",
			ContentPath: "spec.content",
			BaseKind:    "Pipeline",
			Table:       "pipeline_draft_unmarshalled",
			UIDCol:      "draft_uid",
			Fields: []storage.ContentIndexField{
				{Path: "spec.content.metadata.name", Column: "name"},
				{Path: "spec.content.spec.type", Column: "type"},
				// no state: draft's sidecar is intentionally a subset.
			},
		},
	}
}

// fakePlainObject implements runtime.Object but not ContentIndexDescribable,
// proving BuildContentIndexFromScheme skips types that don't opt in instead of
// panicking or misinterpreting them.
type fakePlainObject struct {
	metav1.TypeMeta
}

func (f *fakePlainObject) DeepCopyObject() runtime.Object {
	cp := *f
	return &cp
}

// TestBuildContentIndexFromScheme_AggregatesDescribableTypes proves the runtime
// side of content indexing: given a scheme containing a type that implements
// storage.ContentIndexDescribable, BuildContentIndexFromScheme collects its
// specs (without re-parsing any proto annotation itself) and produces the same
// read-map shape BuildContentIndex would from those specs directly. A
// non-describable type in the same scheme is silently skipped.
func TestBuildContentIndexFromScheme_AggregatesDescribableTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	pipelineGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Pipeline"}
	plainGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Plain"}
	scheme.AddKnownTypeWithName(pipelineGVK, &fakeRevisionedPipeline{})
	scheme.AddKnownTypeWithName(plainGVK, &fakePlainObject{})

	ci := BuildContentIndexFromScheme(scheme)
	require.NotNil(t, ci)

	revisionGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Revision"}
	draftGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Draft"}

	require.Contains(t, ci.ReadMaps[revisionGVK], "spec.content.status.state")
	require.Contains(t, ci.ReadMaps[draftGVK], "spec.content.metadata.name")
	require.NotContains(t, ci.ReadMaps[draftGVK], "spec.content.status.state",
		"draft's sidecar is a subset of revision's — state must not leak in")

	require.Contains(t, ci.WriteSpecs, revisionGVK)
	require.Contains(t, ci.WriteSpecs, draftGVK)
}

// TestBuildContentIndexFromScheme_NoDescribableTypes proves a scheme with no
// content-indexed types (or no scheme at all) leaves the storage in plain mode.
func TestBuildContentIndexFromScheme_NoDescribableTypes(t *testing.T) {
	require.Nil(t, BuildContentIndexFromScheme(nil))

	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(schema.GroupVersionKind{Group: "g", Version: "v", Kind: "Plain"}, &fakePlainObject{})
	require.Nil(t, BuildContentIndexFromScheme(scheme))
}
