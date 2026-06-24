package mysql

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// TestCrossJoinContentSpecs proves the base x revisioned_in expansion: a base
// type emits one spec per wrapper kind it lists in revisioned_in, with the
// wrapper GVK resolved by convention, the table/uid derived from the base kind
// and wrapper kind, and — crucially — each spec's columns coming from that
// wrapper's own content_index list. The two wrappers carry asymmetric subsets to
// prove per-wrapper column selection (revision indexes status.state; draft does
// not).
func TestCrossJoinContentSpecs(t *testing.T) {
	revisionGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Revision"}
	draftGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Draft"}
	pipelineGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Pipeline"}

	bases := []contentBaseInfo{
		{
			gvk: pipelineGVK,
			wrappers: []wrapperContentInfo{
				{
					kind: "revision",
					fields: []indexField{
						{path: "metadata.name", key: "name"},
						{path: "spec.type", key: "type"},
						{path: "status.state", key: "state"},
					},
				},
				{
					kind: "draft",
					fields: []indexField{
						{path: "metadata.name", key: "name"},
						{path: "spec.type", key: "type"},
						// no state: draft's sidecar is intentionally a subset.
					},
				},
			},
		},
	}

	specs := crossJoinContentSpecs(bases)

	// One spec per wrapper entry.
	require.Len(t, specs, 2)

	byKind := map[schema.GroupVersionKind]ContentIndexFieldSpec{}
	for _, s := range specs {
		byKind[s.WrapperGVK] = s
	}

	rev := byKind[revisionGVK]
	require.Equal(t, "spec.content", rev.ContentPath)
	require.Equal(t, "revision", rev.WrapperKind)
	require.Equal(t, "Pipeline", rev.BaseKind)
	require.Equal(t, "pipeline_revision_unmarshalled", rev.Table)
	require.Equal(t, "revision_uid", rev.UIDCol)
	require.Equal(t, []ContentIndexField{
		{Path: "spec.content.metadata.name", Column: "name"},
		{Path: "spec.content.spec.type", Column: "type"},
		{Path: "spec.content.status.state", Column: "state"},
	}, rev.Fields)

	draft := byKind[draftGVK]
	require.Equal(t, "draft", draft.WrapperKind)
	require.Equal(t, "pipeline_draft_unmarshalled", draft.Table)
	require.Equal(t, "draft_uid", draft.UIDCol)
	// Draft carries the asymmetric subset: name + type, but NOT state.
	require.Equal(t, []ContentIndexField{
		{Path: "spec.content.metadata.name", Column: "name"},
		{Path: "spec.content.spec.type", Column: "type"},
	}, draft.Fields)
}
