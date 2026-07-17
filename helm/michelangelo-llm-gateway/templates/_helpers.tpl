{{- define "michelangelo-llm-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "michelangelo-llm-gateway.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "michelangelo-llm-gateway.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end }}

{{- define "michelangelo-llm-gateway.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "michelangelo-llm-gateway.labels" -}}
helm.sh/chart: {{ include "michelangelo-llm-gateway.chart" . }}
{{ include "michelangelo-llm-gateway.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "michelangelo-llm-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "michelangelo-llm-gateway.name" . | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end }}

{{- define "michelangelo-llm-gateway.serverSelectorLabels" -}}
{{ include "michelangelo-llm-gateway.selectorLabels" . }}
app.kubernetes.io/component: server
{{- end }}

{{- define "michelangelo-llm-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "michelangelo-llm-gateway.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end }}

{{- define "michelangelo-llm-gateway.configMapName" -}}
{{- if .Values.config.create -}}
{{- printf "%s-config" (include "michelangelo-llm-gateway.fullname" .) -}}
{{- else -}}
{{- .Values.config.existingConfigMap -}}
{{- end -}}
{{- end }}

{{- define "michelangelo-llm-gateway.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}
