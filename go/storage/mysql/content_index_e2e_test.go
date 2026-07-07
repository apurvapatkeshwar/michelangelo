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
// (as BuildContentIndexFromScheme would produce), it derives a Revision's sidecar
// row by decoding its content blob and calling Pipeline's generated content
// extractor (GetContentIndexedKeyValuePairs).
//
// This uses BuildContentIndex with explicit specs, isolating the write-path
// assertions (decode content -> call the generated extractor -> build a row) from
// how the ContentIndex was assembled. TestBuildContentIndexFromScheme_RealPipelineType
// below covers that assembly step against the same real types.
func TestContentIndexWritePathRealTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(v2.GroupVersion, &v2.Pipeline{}, &v2.PipelineList{}, &v2.Revision{}, &v2.RevisionList{})
	revisionGVK := v2.GroupVersion.WithKind("Revision")

	// The ContentIndex as codegen would emit it, matching pipeline.proto's
	// revisioned_in: pipeline_type, owner, and the content-only state field.
	ci := BuildContentIndex([]ContentIndexFieldSpec{{
		WrapperGVK:  revisionGVK,
		WrapperKind: "revision",
		ContentPath: "spec.content",
		BaseKind:    "Pipeline",
		Table:       "pipeline_revision_unmarshalled",
		UIDCol:      "revision_uid",
		Fields: []ContentIndexField{
			{Path: "spec.content.spec.type", Column: "pipeline_type"},
			{Path: "spec.content.spec.owner.name", Column: "owner"},
			{Path: "spec.content.status.state", Column: "state"},
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

// TestBuildContentIndexFromScheme_RealPipelineType is the strongest proof that
// codegen and the runtime routing agree: unlike TestContentIndexWritePathRealTypes
// above (which deliberately hand-builds a ContentIndex to stay independent of
// this mechanism), this calls BuildContentIndexFromScheme against a scheme
// containing the REAL, freshly-generated v2.Pipeline. Pipeline.ContentIndexFieldSpecs()
// is generated code, not a test fixture — if codegen and this function ever
// disagreed, this test would be the one to catch it.
func TestBuildContentIndexFromScheme_RealPipelineType(t *testing.T) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(v2.GroupVersion, &v2.Pipeline{}, &v2.PipelineList{}, &v2.Revision{}, &v2.RevisionList{})

	ci := BuildContentIndexFromScheme(scheme)
	require.NotNil(t, ci)

	revisionGVK := v2.GroupVersion.WithKind("Revision")
	readMap := ci.ReadMaps[revisionGVK]
	require.NotNil(t, readMap, "Pipeline's revisioned_in should route through the real Revision GVK")

	require.Equal(t,
		[]contentIndexEntry{{BaseType: "Pipeline", Table: "pipeline_revision_unmarshalled", Column: "pipeline_type", UIDCol: "revision_uid"}},
		readMap["spec.content.spec.type"],
	)
	require.Equal(t,
		[]contentIndexEntry{{BaseType: "Pipeline", Table: "pipeline_revision_unmarshalled", Column: "owner", UIDCol: "revision_uid"}},
		readMap["spec.content.spec.owner.name"],
	)
	require.Equal(t,
		[]contentIndexEntry{{BaseType: "Pipeline", Table: "pipeline_revision_unmarshalled", Column: "state", UIDCol: "revision_uid"}},
		readMap["spec.content.status.state"],
	)

	require.Contains(t, ci.WriteSpecs, revisionGVK, "write routing for Revision must also be populated")
}

