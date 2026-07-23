output "endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "password" {
  value = random_password.db.result
  sensitive = true
}

output "id" {
  value = aws_db_instance.postgres.id
}
