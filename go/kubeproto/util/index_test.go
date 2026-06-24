package util_test

import (
	"testing"

	api "github.com/michelangelo-ai/michelangelo/proto-go/api"
	testpb "github.com/michelangelo-ai/michelangelo/proto-go/test/kubeproto"
	testerrpb "github.com/michelangelo-ai/michelangelo/proto-go/test/kubeproto/indexing_errors"

	"github.com/michelangelo-ai/michelangelo/go/kubeproto/pboptions"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
)

func readInput(t *testing.T, reqData []byte) (*protogen.Plugin, *protoregistry.Types) {
	gen, extTypes, err := util.GetPluginAndExtensions(reqData, true)
	assert.NoError(t, err)

	return gen, extTypes
}

func TestParseIndexedFields(t *testing.T) {
	tests := map[string][]util.IndexedField{
		"TestIndexing": {
			{
				Key:       "key01",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "Name"},
				ProtoPath: "spec.name",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key02",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Status", "Count"},
				ProtoPath: "status.count",
				Type:      "INT",
			},
			{
				Key:       "key03",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedInt32"},
				ProtoPath: "spec.sample_msg.indexed_int32",
				Type:      "INT",
			},
			{
				Key:       "key04",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedInt64"},
				ProtoPath: "spec.sample_msg.indexed_int64",
				Type:      "BIGINT",
			},
			{
				Key:       "key05",
				Flag:      util.IndexFlagCompositeKey,
				GoPaths:   []string{"Spec", "IndexedResourceId"},
				ProtoPath: "spec.indexed_resource_id",
				SubFields: []util.IndexedSubField{
					{
						Key:       "key05_namespace",
						GoPath:    "Namespace",
						ProtoPath: "spec.indexed_resource_id.namespace",
						Type:      "VARCHAR(255)",
					},
					{
						Key:       "key05_name",
						GoPath:    "Name",
						ProtoPath: "spec.indexed_resource_id.name",
						Type:      "VARCHAR(255)",
					},
				},
			},
			{
				Key:       "key06",
				GoPaths:   []string{"Spec", "IndexedUserInfo"},
				ProtoPath: "spec.indexed_user_info",
				SubFields: []util.IndexedSubField{
					{
						Key:       "key06_name",
						GoPath:    "Name",
						ProtoPath: "spec.indexed_user_info.name",
						Type:      "VARCHAR(255)",
					},
					{
						Key:       "key06_proxy_user",
						GoPath:    "ProxyUser",
						ProtoPath: "spec.indexed_user_info.proxy_user",
						Type:      "VARCHAR(255)",
					},
				},
			},
			{
				Key:       "key07",
				Flag:      util.IndexFlagPrimitive | util.IndexFlagEnum,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedEnum"},
				ProtoPath: "spec.sample_msg.indexed_enum",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key08",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfStr()"},
				ProtoPath: "spec.one_of_str",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key09",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfInt64()"},
				ProtoPath: "spec.one_of_int64",
				Type:      "BIGINT",
			},
			{
				Key:       "key10",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfSampleMsg()", "IndexedInt32"},
				ProtoPath: "spec.one_of_sample_msg.indexed_int32",
				Type:      "INT",
			},
			{
				Key:       "key11",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "GetOneOfStr()"},
				ProtoPath: "spec.sample_msg.one_of_str",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key12",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfSampleMsg()", "GetOneOfInt64()"},
				ProtoPath: "spec.one_of_sample_msg.one_of_int64",
				Type:      "BIGINT",
			},
			{
				Key:       "key13",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Status", "TimeMsg"},
				ProtoPath: "status.time_msg",
				Type:      "DATETIME",
			},
			{
				Key:       "key14",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Metadata", "DeletionTimestamp"},
				ProtoPath: "metadata.deletionTimestamp",
				Type:      "DATETIME",
			},
			{
				Key:       "key15",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"TypeMeta", "Kind"},
				ProtoPath: "type_meta.kind",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key16",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedBool"},
				ProtoPath: "spec.sample_msg.indexed_bool",
				Type:      "BOOLEAN",
			},
			{
				Key:       "key17",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfSampleMsg()", "IndexedBool"},
				ProtoPath: "spec.one_of_sample_msg.indexed_bool",
				Type:      "BOOLEAN",
			},
			{
				Key:       "key18",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedString"},
				ProtoPath: "spec.sample_msg.indexed_string",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key19",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "SampleMsg", "IndexedLongString"},
				ProtoPath: "spec.sample_msg.indexed_long_string",
				Type:      "VARCHAR(768)",
			},
			{
				Key:       "key20",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfSampleMsg()", "IndexedString"},
				ProtoPath: "spec.one_of_sample_msg.indexed_string",
				Type:      "VARCHAR(255)",
			},
			{
				Key:       "key21",
				Flag:      util.IndexFlagPrimitive,
				GoPaths:   []string{"Spec", "GetOneOfSampleMsg()", "IndexedLongString"},
				ProtoPath: "spec.one_of_sample_msg.indexed_long_string",
				Type:      "VARCHAR(768)",
			},
		},
		"TestMsg3":   nil,
		"TestObject": nil,
	}
	tested := 0
	gen, extTypes := readInput(t, testpb.GetProtocReqData())

	for _, f := range gen.Files {
		if !f.Generate {
			continue
		}

		for _, msg := range f.Messages {
			pbOptions := msg.Desc.Options().(*descriptorpb.MessageOptions)
			options, err := pboptions.ReadOptions(extTypes, pbOptions)
			assert.Nil(t, err)

			if options.Bool("has_resource") {
				indexedFields := util.ParseIndexedFields(msg, options)
				if expectedResult, ok := tests[msg.GoIdent.GoName]; ok {
					assert.Equal(t, expectedResult, indexedFields)
					tested++
				}
			}
		}
	}

	assert.Equal(t, len(tests), tested)
}

func TestParseIndexedFieldsErrors(t *testing.T) {
	tests := map[string]string{
		"TestIndexingInvalidPath":   "Invalid path in index annotation. key: key01, path: .spec.name",
		"TestIndexingDuplicatedKey": "Invalid index annotation. Duplicated key. key: key01, path: spec.int32_field",
	}
	tested := 0
	gen, extTypes := readInput(t, testerrpb.GetProtocReqData())

	for _, f := range gen.Files {
		if !f.Generate {
			continue
		}

		for _, msg := range f.Messages {
			pbOptions := msg.Desc.Options().(*descriptorpb.MessageOptions)
			options, err := pboptions.ReadOptions(extTypes, pbOptions)
			assert.Nil(t, err)

			if options.Bool("has_resource") {
				if panicMsg, shouldPanic := tests[msg.GoIdent.GoName]; shouldPanic {
					assertPanic(t, panicMsg, func() {
						util.ParseIndexedFields(msg, options)
					})
					tested++
					continue
				}
			}
		}
	}

	assert.Equal(t, len(tests), tested)
}

// TestGeneratedContentIndexedKeyValuePairs exercises the generated content
// extractor (GetContentIndexedKeyValuePairs) on the real generated MiniBase type.
// MiniBase opts into "mini_wrapper" only and its content_index declares a
// primitive (spec.name -> mini_name) and a composite (spec.ref -> the
// ResourceIdentifier's namespace+name subfields), so this asserts: per-wrapper
// keying, primitive extraction, composite subfield expansion, and that a wrapper
// the base did NOT opt into yields no entry.
func TestGeneratedContentIndexedKeyValuePairs(t *testing.T) {
	mb := &testpb.MiniBase{
		Spec: testpb.MiniBaseSpec{
			Name: "m1",
			Ref:  &api.ResourceIdentifier{Namespace: "ns", Name: "r1"},
		},
	}

	result := mb.GetContentIndexedKeyValuePairs()

	// Only the opted-in wrapper kind appears.
	_, hasDraft := result["mini_draft"]
	assert.False(t, hasDraft, "mini_draft was not opted into via revisioned_in")
	fields, ok := result["mini_wrapper"]
	assert.True(t, ok, "mini_wrapper sidecar columns should be present")

	cols := map[string]interface{}{}
	for _, f := range fields {
		cols[f.Key] = f.Value
	}
	assert.Equal(t, "m1", cols["mini_name"])          // primitive
	assert.Equal(t, "ns", cols["mini_ref_namespace"]) // composite subfield
	assert.Equal(t, "r1", cols["mini_ref_name"])      // composite subfield
}

func assertPanic(t *testing.T, expected interface{}, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic but got none")
		assert.Equal(t, expected, r, "unexpected panic message")
	}()
	f()
}
