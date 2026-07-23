resource "random_password" "db" {
  length  = 24
  special = false
}

resource "aws_security_group" "rds" {
  name        = "gpuopt-${var.environment}-rds"
  description = "Security group for GPUOpt RDS PostgreSQL"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.eks_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_db_subnet_group" "rds" {
  name        = "gpuopt-${var.environment}"
  description = "GPUOpt RDS subnet group"
  subnet_ids  = var.private_subnet_ids
  tags        = var.tags
}

resource "aws_db_instance" "postgres" {
  identifier = "gpuopt-${var.environment}"

  engine         = "postgres"
  engine_version = "16.3"
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  db_name               = "gpuopt"
  username              = "gpuopt"
  password              = random_password.db.result
  port                  = 5432
  parameter_group_family = "postgres16"

  db_subnet_group_name   = aws_db_subnet_group.rds.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = var.environment == "prod" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  skip_final_snapshot = var.environment != "prod"
  deletion_protection = var.environment == "prod"

  tags = var.tags
}
