resource "kubernetes_namespace" "gpuopt" {
  metadata {
    name = "gpuopt-system"
    labels = {
      "app.kubernetes.io/part-of" = "gpuopt"
    }
  }
}

resource "kubernetes_secret" "gpuopt_db" {
  count = var.database_type == "postgres" ? 1 : 0
  metadata {
    name      = "gpuopt-db"
    namespace = kubernetes_namespace.gpuopt.metadata[0].name
  }
  data = {
    "database-url" = var.database_url
  }
}

resource "helm_release" "gpuopt" {
  name       = "gpuopt-${var.environment}"
  namespace  = kubernetes_namespace.gpuopt.metadata[0].name
  repository = "oci://ghcr.io/024-code/charts"
  chart      = "gpuopt"
  version    = "0.1.0"

  set {
    name  = "replicaCount"
    value = var.replica_count
  }

  set {
    name  = "image.repository"
    value = var.image_repository
  }

  set {
    name  = "image.tag"
    value = var.image_tag
  }

  set {
    name  = "database.type"
    value = var.database_type
  }

  set {
    name  = "env.GPUOPT_ENV"
    value = var.environment == "prod" ? "production" : "development"
  }

  set {
    name  = "ingress.enabled"
    value = var.domain_name != ""
  }

  dynamic "set" {
    for_each = var.domain_name != "" ? [1] : []
    content {
      name  = "ingress.hosts[0].host"
      value = var.domain_name
    }
  }

  depends_on = [kubernetes_namespace.gpuopt]
}
