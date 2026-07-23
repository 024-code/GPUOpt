output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_ca_certificate" {
  description = "EKS cluster CA certificate"
  value       = module.eks.cluster_ca_certificate
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_password" {
  description = "RDS PostgreSQL password"
  value       = module.rds.password
  sensitive   = true
}

output "gpuopt_url" {
  description = "GPUOpt application URL"
  value       = module.eks.gpuopt_url
}

output "helm_release_name" {
  description = "Helm release name for GPUOpt"
  value       = module.helm.release_name
}
