package main

import (
	"bytes"
	"log"
	"os"
	"strings"

	"github.com/michelangelo-ai/michelangelo/go/api/utils"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/pboptions"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/templates"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"

	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

var logger = log.New(os.Stderr, "", 0)

func getIndexName(tableName, key string) string {
	return tableName + "_" + key
}

func generateSQLSchema(crdRootMsg *protogen.Message, crdOptions *pboptions.Options, msgByName map[protoreflect.FullName]*protogen.Message) []byte {
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

	// Generate content_index sidecar ("*_unmarshalled") tables — one per
	// content_index annotation, holding the columns extracted from the wrapped
	// google.protobuf.Any content so the framework can filter on them via JOIN.
	for _, ci := range util.ParseContentIndexedFields(crdOptions, msgByName) {
		emitUnmarshalledTable(&buf, crdTableName, ci)
	}
	return buf.Bytes()
}

// emitUnmarshalledTable writes one content_index sidecar table:
//
//	CREATE TABLE `<base_type>_<crd>_unmarshalled` (
//	    `<crd>_uid`  VARCHAR(255) NOT NULL,
//	    `<key>`      <type>, ...
//	    PRIMARY KEY (`<crd>_uid`),
//	    KEY `..._<key>` (`<key>`), ...
//	);
func emitUnmarshalledTable(buf *bytes.Buffer, crdTableName string, ci util.ContentIndex) {
	tableName := utils.ToSnakeCase(ci.BaseTypeShortName) + "_" + crdTableName + "_unmarshalled"
	uidColumn := crdTableName + "_uid"

	templates.CRDMySQLUnmarshalledTable.Execute(buf, struct {
		TableName string
		UIDColumn string
	}{tableName, uidColumn})

	// Indexed columns (primitive fields directly; composite message fields as
	// one column per subfield) — same shape as the main table's indexed columns.
	for _, field := range ci.Fields {
		if field.Flag&util.IndexFlagPrimitive != 0 {
			buf.Write([]byte("    `" + field.Key + "`    " + field.Type + ",\n"))
		} else {
			for _, subField := range field.SubFields {
				buf.Write([]byte("    `" + subField.Key + "`    " + subField.Type + ",\n"))
			}
		}
	}

	buf.Write([]byte("    PRIMARY KEY (`" + uidColumn + "`)"))
	for _, field := range ci.Fields {
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

	// Index every message across all files (generated + imported) by full name,
	// so content_index annotations can resolve their base_type to a descriptor.
	msgByName := make(map[protoreflect.FullName]*protogen.Message)
	for _, f := range gen.Files {
		registerMessages(msgByName, f.Messages)
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
				buf = generateSQLSchema(msg, options, msgByName)
			}
		}

		_, err = g.Write(buf)
		if err != nil {
			logger.Panicf("failed to write to generated file: %v", err)
		}
	}

	return gen.Response()
}

// registerMessages records each message (and its nested messages) by full name.
func registerMessages(byName map[protoreflect.FullName]*protogen.Message, msgs []*protogen.Message) {
	for _, msg := range msgs {
		byName[msg.Desc.FullName()] = msg
		registerMessages(byName, msg.Messages)
	}
}

func main() {
	reqData := util.ReadRequest()
	resp := generateSQL(reqData)
	util.WriteResponse(resp)
}
