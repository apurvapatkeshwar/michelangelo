package mysql

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestContentIndexUpsertSQL(t *testing.T) {
	row := ContentIndexRow{
		Table:  "pipeline_revision_unmarshalled",
		UIDCol: "revision_uid",
		UID:    "uid-1",
		Columns: []contentIndexColumn{
			{Name: "pipeline_type", Value: "PIPELINE_TYPE_TRAIN"},
			{Name: "owner", Value: "bob"},
		},
	}
	query, args := contentIndexUpsertSQL(row)
	require.Equal(t,
		"INSERT INTO `pipeline_revision_unmarshalled` (`revision_uid`, `pipeline_type`, `owner`) "+
			"VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE `pipeline_type` = VALUES(`pipeline_type`), "+
			"`owner` = VALUES(`owner`)",
		query,
	)
	require.Equal(t, []interface{}{"uid-1", "PIPELINE_TYPE_TRAIN", "bob"}, args)
}

// TestBuildContentIndex proves the single field-spec input derives both the read
// map (path -> sidecar entry, for filtering) and the write spec (base kind ->
// target table, for population) — so reads and writes can't disagree.
func TestBuildContentIndex(t *testing.T) {
	revisionGVK := schema.GroupVersionKind{Group: "michelangelo.api", Version: "v2beta1", Kind: "Revision"}
	ci := BuildContentIndex([]ContentIndexFieldSpec{
		{
			WrapperGVK:  revisionGVK,
			ContentPath: "spec.content",
			BaseKind:    "Pipeline",
			Table:       "pipeline_revision_unmarshalled",
			UIDCol:      "revision_uid",
			Fields: []ContentIndexField{
				{Path: "spec.content.spec.type", Column: "pipeline_type"},
				{Path: "spec.content.spec.owner.name", Column: "owner"},
			},
		},
	})

	// Read map: each criterion path resolves to its sidecar entry.
	readMap := ci.ReadMaps[revisionGVK]
	require.Equal(t,
		[]contentIndexEntry{{BaseType: "Pipeline", Table: "pipeline_revision_unmarshalled", Column: "pipeline_type", UIDCol: "revision_uid"}},
		readMap["spec.content.spec.type"],
	)
	require.Equal(t,
		[]contentIndexEntry{{BaseType: "Pipeline", Table: "pipeline_revision_unmarshalled", Column: "owner", UIDCol: "revision_uid"}},
		readMap["spec.content.spec.owner.name"],
	)

	// Write spec: the content path and the per-base-kind target table, including
	// the per-column extract paths (the spec.content prefix stripped) the write
	// path navigates over the decoded base message.
	writeSpec := ci.WriteSpecs[revisionGVK]
	require.Equal(t, "spec.content", writeSpec.contentPath)
	require.Equal(t,
		contentIndexTarget{
			table:  "pipeline_revision_unmarshalled",
			uidCol: "revision_uid",
			fields: []contentExtractField{
				{contentPath: "spec.type", column: "pipeline_type"},
				{contentPath: "spec.owner.name", column: "owner"},
			},
		},
		writeSpec.targets["Pipeline"],
	)
}
