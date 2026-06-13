package main

import (
	"bytes"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/michelangelo-ai/michelangelo/go/api/utils"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/pboptions"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/templates"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"

	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

var logger = log.New(os.Stderr, "", 0)

// revisionedBaseType is a CRD with a non-empty resource.revisioned_in: a base
// resource snapshotted into the listed content-wrapper kinds (e.g. "revision",
// "draft"). Its own michelangelo.api.index annotations are inherited by each
// matching wrapper's sidecar table so there is no separate field list to maintain.
type revisionedBaseType struct {
	// shortName is the CRD message's Go name (e.g. "Pipeline"), snake-cased into
	// the sidecar table name.
	shortName string
	// wrapperKinds is the set of wrapper kinds this base type opts into (from
	// resource.revisioned_in). A sidecar is emitted only for wrappers whose
	// content_wrapper.kind is in this set.
	wrapperKinds map[string]bool
	// fields are the base type's indexed fields — the same set its own main table uses.
	fields []util.IndexedField
}

func getIndexName(tableName, key string) string {
	return tableName + "_" + key
}

func generateSQLSchema(crdRootMsg *protogen.Message, crdOptions *pboptions.Options, revisionedBaseTypes []revisionedBaseType) []byte {
	var buf bytes.Buffer
	indexedFields := util.ParseIndexedFields(crdRootMsg, crdOptions)
	crdName := strings.ToUpper(crdRootMsg.GoIdent.GoName[:1]) + crdRootMsg.GoIdent.GoName[1:]
	crdTableName := utils.ToSnakeCase(crdName)

	// Generate main table
	typeInfo := struct {
		TableName string
	}{crdTableName}
	templates.CRDMySQLMainTableColumn.Execute(&buf, typeInfo)

	// Generate CRD specified indexed columns
	for _, field := range indexedFields {
		if field.Flag&util.IndexFlagPrimitive != 0 {
			buf.Write([]byte("    `" + field.Key + "`    " + field.Type + ",\n"))
		} else {
			for _, subField := range field.SubFields {
				buf.Write([]byte("    `" + subField.Key + "`    " + subField.Type + ",\n"))
			}

		}
	}

	templates.CRDMySQLMainTableIndex.Execute(&buf, typeInfo)

	// Generate CRD specified indexes
	for _, field := range indexedFields {
		buf.Write([]byte(",\n"))
		if field.Flag&util.IndexFlagPrimitive != 0 {
			buf.Write([]byte("    KEY    `" + getIndexName(crdTableName, field.Key) + "` (`" + field.Key + "`)"))
		} else {
			if field.Flag&util.IndexFlagCompositeKey != 0 {
				buf.Write([]byte("    KEY    `" + getIndexName(crdTableName, field.Key) + "` ("))
				firstSubfield := true
				for _, subField := range field.SubFields {
					if firstSubfield {
						firstSubfield = false
					} else {
						buf.Write([]byte(", "))
					}
					buf.Write([]byte("`" + subField.Key + "`"))
				}
				buf.Write([]byte(")"))
			} else {
				firstSubField := true
				for _, subField := range field.SubFields {
					if firstSubField {
						firstSubField = false
					} else {
						buf.Write([]byte(",\n"))
					}
					buf.Write([]byte("    KEY    `" + getIndexName(crdTableName, subField.Key) + "` (`" + subField.Key + "`)"))
				}
			}
		}
	}
	buf.Write([]byte("\n);"))

	templates.CRDMySQLLabelAnnotationTable.Execute(&buf, typeInfo)

	// Content wrappers (marked content_wrapper) get one sidecar
	// "<base>_<wrapper>_unmarshalled" table per revisioned base type that opted
	// this wrapper's kind into its revisioned_in. Columns are inherited from the
	// base type's own index annotations, so the wrapper and base type never have
	// to name each other — they only share the wrapper-kind string.
	if crdOptions.Bool("has_content_wrapper") {
		wrapperKind := crdOptions.String("content_wrapper.kind")
		for _, base := range revisionedBaseTypes {
			if base.wrapperKinds[wrapperKind] {
				emitUnmarshalledTable(&buf, crdTableName, base)
			}
		}
	}
	return buf.Bytes()
}

// emitUnmarshalledTable writes one content sidecar table for a (wrapper, base)
// pair, inheriting the base type's indexed columns:
//
//	CREATE TABLE `<base>_<wrapper>_unmarshalled` (
//	    `<wrapper>_uid`  VARCHAR(255) NOT NULL,
//	    `<key>`      <type>, ...
//	    PRIMARY KEY (`<wrapper>_uid`),
//	    KEY `..._<key>` (`<key>`), ...
//	);
func emitUnmarshalledTable(buf *bytes.Buffer, wrapperTableName string, base revisionedBaseType) {
	tableName := utils.ToSnakeCase(base.shortName) + "_" + wrapperTableName + "_unmarshalled"
	uidColumn := wrapperTableName + "_uid"

	templates.CRDMySQLUnmarshalledTable.Execute(buf, struct {
		TableName string
		UIDColumn string
	}{tableName, uidColumn})

	// Indexed columns (primitive fields directly; composite message fields as one
	// column per subfield) — same shape as the base type's own main-table columns.
	for _, field := range base.fields {
		if field.Flag&util.IndexFlagPrimitive != 0 {
			buf.Write([]byte("    `" + field.Key + "`    " + field.Type + ",\n"))
		} else {
			for _, subField := range field.SubFields {
				buf.Write([]byte("    `" + subField.Key + "`    " + subField.Type + ",\n"))
			}
		}
	}

	buf.Write([]byte("    PRIMARY KEY (`" + uidColumn + "`)"))
	for _, field := range base.fields {
		if field.Flag&util.IndexFlagPrimitive != 0 {
			buf.Write([]byte(",\n    KEY    `" + getIndexName(tableName, field.Key) + "` (`" + field.Key + "`)"))
		} else {
			for _, subField := range field.SubFields {
				buf.Write([]byte(",\n    KEY    `" + getIndexName(tableName, subField.Key) + "` (`" + subField.Key + "`)"))
			}
		}
	}
	buf.Write([]byte("\n);\n"))
}

func generateSQL(reqData []byte) *pluginpb.CodeGeneratorResponse {
	gen, extTypes, err := util.GetPluginAndExtensions(reqData, true)
	if err != nil {
		logger.Panic(err)
	}

	// Collect every revisioned base type across all files (generated + imported),
	// so each content wrapper can emit a sidecar table per base type inheriting
	// that base type's own index annotations.
	var revisionedBaseTypes []revisionedBaseType
	for _, f := range gen.Files {
		collectRevisionedBaseTypes(&revisionedBaseTypes, extTypes, f.Messages)
	}

	for _, f := range gen.Files {
		// Skip the proto file that don't need to generate go code,
		// such as imported proto files.
		if !f.Generate {
			continue
		}

		filename := f.GeneratedFilenamePrefix + ".pb.sql"
		g := gen.NewGeneratedFile(filename, f.GoImportPath)
		var buf []byte
		for _, msg := range f.Messages {
			pbOptions := msg.Desc.Options().(*descriptorpb.MessageOptions)
			options, e := pboptions.ReadOptions(extTypes, pbOptions)
			if e != nil {
				logger.Panicf("Failed to parse the options of message %v: %v", msg.GoIdent.GoName, e)
			}

			if options.Bool("has_resource") {
				buf = append(buf, generateSQLSchema(msg, options, revisionedBaseTypes)...)
			}
		}

		_, err = g.Write(buf)
		if err != nil {
			logger.Panicf("failed to write to generated file: %v", err)
		}
	}

	return gen.Response()
}

// collectRevisionedBaseTypes appends every message (recursively) with a non-empty
// resource.revisioned_in, recording which wrapper kinds it opts into and parsing
// its index fields so matching wrapper sidecar tables can inherit them.
func collectRevisionedBaseTypes(out *[]revisionedBaseType, extTypes *protoregistry.Types, msgs []*protogen.Message) {
	for _, msg := range msgs {
		pbOptions := msg.Desc.Options().(*descriptorpb.MessageOptions)
		options, err := pboptions.ReadOptions(extTypes, pbOptions)
		if err != nil {
			logger.Panicf("Failed to parse the options of message %v: %v", msg.GoIdent.GoName, err)
		}
		if kinds := readRevisionedIn(options); len(kinds) > 0 {
			*out = append(*out, revisionedBaseType{
				shortName:    msg.GoIdent.GoName,
				wrapperKinds: kinds,
				fields:       util.ParseIndexedFields(msg, options),
			})
		}
		collectRevisionedBaseTypes(out, extTypes, msg.Messages)
	}
}

// readRevisionedIn returns the set of wrapper kinds listed in resource.revisioned_in.
func readRevisionedIn(options *pboptions.Options) map[string]bool {
	count := int(options.Int64("resource.len(revisioned_in)"))
	if count == 0 {
		return nil
	}
	kinds := make(map[string]bool, count)
	for i := 0; i < count; i++ {
		kinds[options.String("resource.revisioned_in["+strconv.Itoa(i)+"]")] = true
	}
	return kinds
}

func main() {
	reqData := util.ReadRequest()
	resp := generateSQL(reqData)
	util.WriteResponse(resp)
}
