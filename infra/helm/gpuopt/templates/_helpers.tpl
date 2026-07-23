{{- define "gpuopt.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "gpuopt.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "gpuopt.labels" -}}
helm.sh/chart: {{ include "gpuopt.name" . }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "gpuopt.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: gpuopt
{{- end }}

{{- define "gpuopt.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gpuopt.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "gpuopt.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "gpuopt.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "gpuopt.databaseUrl" -}}
{{- if .Values.database.url }}
{{- .Values.database.url }}
{{- else if eq .Values.database.type "postgres" }}
{{- $host := .Values.database.postgres.host }}
{{- $port := .Values.database.postgres.port }}
{{- $user := .Values.database.postgres.user }}
{{- $pass := .Values.database.postgres.password }}
{{- $db := .Values.database.postgres.database }}
{{- printf "postgresql://%s:%s@%s:%d/%s" $user $pass $host $port $db }}
{{- else }}
{{- printf "sqlite:///%s" .Values.env.GPUOPT_DATABASE_PATH }}
{{- end }}
{{- end }}
