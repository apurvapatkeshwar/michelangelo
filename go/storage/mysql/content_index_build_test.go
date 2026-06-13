package mysql

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// TestCrossJoinContentSpecs proves the (wrapper x revisioned base) cross-join:
// a base that opts a wrapper kind in gets a spec (with derived table/uid/paths),
// and a base that doesn't opt a wrapper kind in is excluded.
func TestCrossJoinContentSpecs(t *testing.T) {
	revisionGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Revision"}
	draftGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Draft"}
	pipelineGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Pipeline"}

	wrappers := []contentWrapperInfo{
		{gvk: revisionGVK, contentPath: "spec.content", kind: "revision"},
		{gvk: draftGVK, contentPath: "spec.content", kind: "draft"},
	}
	bases := []contentBaseInfo{
		{
			gvk:          pipelineGVK,
			revisionedIn: []string{"revision"}, // opts into revision only, NOT draft
			indexFields: []indexField{
				{path: "spec.type", key: "pipeline_type"},
				{path: "status.state", key: "state"},
			},
		},
	}

	specs := crossJoinContentSpecs(wrappers, bases)

	// Pipeline opted into "revision" only -> exactly one spec, for the revision wrapper.
	require.Len(t, specs, 1)
	got := specs[0]
	require.Equal(t, revisionGVK, got.WrapperGVK)
	require.Equal(t, "spec.content", got.ContentPath)
	require.Equal(t, "Pipeline", got.BaseKind)
	require.Equal(t, "pipeline_revision_unmarshalled", got.Table)
	require.Equal(t, "revision_uid", got.UIDCol)
	require.Equal(t, []ContentIndexField{
		{Path: "spec.content.spec.type", Column: "pipeline_type"},
		{Path: "spec.content.status.state", Column: "state"},
	}, got.Fields)
}
