variable "environment" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "image_repository" {
  type = string
}

variable "image_tag" {
  type = string
}

variable "replica_count" {
  type = number
  default = 2
}

variable "database_url" {
  type = string
  default = ""
}

variable "database_type" {
  type = string
  default = "sqlite"
}

variable "domain_name" {
  type = string
  default = ""
}
