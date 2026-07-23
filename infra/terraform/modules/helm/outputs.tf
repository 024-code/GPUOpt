output "release_name" {
  value = helm_release.gpuopt.name
}

output "namespace" {
  value = helm_release.gpuopt.namespace
}
