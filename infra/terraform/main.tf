module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "${var.cluster_name}-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev"
  enable_dns_hostnames = true

  tags = var.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = "${var.cluster_name}-${var.environment}"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = var.environment != "prod"

  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    general = {
      instance_types = var.eks_node_instance_types
      min_size       = var.eks_node_min_size
      max_size       = var.eks_node_max_size
      desired_size   = var.eks_node_desired_size
      subnet_ids     = module.vpc.private_subnets
    }
  }

  tags = var.tags
}

module "eks_gpu_nodes" {
  source  = "terraform-aws-modules/eks/aws//modules/eks-managed-node-group"
  version = "~> 20.8"

  count = var.gpu_node_enabled ? 1 : 0

  name            = "${var.cluster_name}-${var.environment}-gpu"
  cluster_name    = module.eks.cluster_name
  cluster_version = "1.29"

  subnet_ids      = module.vpc.private_subnets
  instance_types  = var.gpu_node_instance_types
  min_size        = var.gpu_node_min_size
  max_size        = var.gpu_node_max_size
  desired_size    = var.gpu_node_desired_size

  create_iam_role = false
  iam_role_arn    = module.eks.eks_managed_node_groups["general"].iam_role_arn

  tags = var.tags
}

module "rds" {
  source = "./modules/rds"
  count  = var.rds_enabled ? 1 : 0

  environment          = var.environment
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnets
  instance_class       = var.rds_instance_class
  allocated_storage    = var.rds_allocated_storage
  eks_security_group_id = module.eks.cluster_security_group_id

  tags = var.tags
}

locals {
  database_url = var.rds_enabled ? "postgresql://gpuopt:${module.rds[0].password}@${module.rds[0].endpoint}/gpuopt" : ""
}

module "helm" {
  source = "./modules/helm"

  environment  = var.environment
  cluster_name = module.eks.cluster_name

  image_repository = var.gpuopt_image_repository
  image_tag        = var.gpuopt_image_tag
  replica_count    = var.gpuopt_replicas
  database_url     = local.database_url
  database_type    = var.rds_enabled ? "postgres" : "sqlite"
  domain_name      = var.domain_name

  depends_on = [module.eks]
}

resource "aws_ecr_repository" "gpuopt" {
  name                 = "gpuopt-backend"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = var.tags
}
