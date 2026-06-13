package mysql

import (
	"testing"

	gogotypes "github.com/gogo/protobuf/types"
	v2 "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// TestContentIndexWritePathRealTypes proves the content-index write-path glue
// against the REAL v2 Pipeline/Revision types (no database): given a ContentIndex
// (as the codegen would emit), deriving a Revision's sidecar row by decoding its
// content blob and reusing Pipeline's own indexed extractor.
//
// NOTE: this deliberately uses BuildContentIndex with explicit specs rather than
// BuildContentIndexFromScheme. Local testing showed BuildContentIndexFromScheme
// can't read content_wrapper/revisioned_in off gogo types at runtime — gogo's
// generated descriptor drops those newly-added custom options (GetExtension
// returns "missing extension"). So the ContentIndex must be produced by codegen,
// not runtime reflection. The plan's read/write logic below is independent of
// that and works on real types.
func TestContentIndexWritePathRealTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(v2.GroupVersion, &v2.Pipeline{}, &v2.PipelineList{}, &v2.Revision{}, &v2.RevisionList{})
	revisionGVK := v2.GroupVersion.WithKind("Revision")

	// The ContentIndex as codegen would emit it. The real v2 Pipeline currently
	// indexes only owner + pipeline_type, so those are the sidecar columns.
	ci := BuildContentIndex([]ContentIndexFieldSpec{{
		WrapperGVK:  revisionGVK,
		ContentPath: "spec.content",
		BaseKind:    "Pipeline",
		Table:       "pipeline_revision_unmarshalled",
		UIDCol:      "revision_uid",
		Fields: []ContentIndexField{
			{Path: "spec.content.spec.type", Column: "pipeline_type"},
			{Path: "spec.content.spec.owner.name", Column: "owner"},
		},
	}})

	// A real Revision whose content is a real Pipeline.
	pipeline := &v2.Pipeline{
		Spec: v2.PipelineSpec{
			Type:  v2.PIPELINE_TYPE_TRAIN,
			Owner: &v2.UserInfo{Name: "bob"},
		},
	}
	content, err := gogotypes.MarshalAny(pipeline)
	require.NoError(t, err)

	rev := &v2.Revision{
		TypeMeta:   metav1.TypeMeta{Kind: "Revision", APIVersion: v2.GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{UID: "rev-1", Namespace: "ns", Name: "p1-rev1"},
		Spec: v2.RevisionSpec{
			BaseType: &metav1.TypeMeta{Kind: "Pipeline"},
			Content:  content,
		},
	}

	m := &mysqlMetadataStorage{
		scheme:                 scheme,
		contentIndexMaps:       ci.ReadMaps,
		contentIndexWriteSpecs: ci.WriteSpecs,
	}

	// Write path: decode the revision's content -> reuse Pipeline's extractor ->
	// build the sidecar row.
	rows, err := m.contentIndexRows(rev)
	require.NoError(t, err)
	require.Len(t, rows, 1, "expected one sidecar row for a Pipeline revision")

	row := rows[0]
	require.Equal(t, "pipeline_revision_unmarshalled", row.Table)
	require.Equal(t, "revision_uid", row.UIDCol)
	require.Equal(t, "rev-1", row.UID)

	cols := map[string]interface{}{}
	for _, c := range row.Columns {
		cols[c.Name] = c.Value
	}
	// Values came from the wrapped Pipeline inside content (not any pipeline store).
	require.Equal(t, "bob", cols["owner"])
	require.Contains(t, cols, "pipeline_type")

	// Read path: the upsert SQL for that row is well-formed.
	q, args := contentIndexUpsertSQL(row)
	require.Contains(t, q, "INSERT INTO `pipeline_revision_unmarshalled`")
	require.Contains(t, q, "ON DUPLICATE KEY UPDATE")
	require.Equal(t, "rev-1", args[0])
}
