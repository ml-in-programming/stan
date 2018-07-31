provider "aws" {
  region = "eu-west-1"
  profile = "ml-labs-jetbrains"
}

terraform {
  backend "s3" {
    bucket = "terraform-state.ml.aws.intellij.net"
    key = "stan/terraform.tfstate"
    profile = "ml-labs-jetbrains"
    region = "eu-west-1"
  }
  required_version = "0.11.7"
}

locals {
  region = "eu-west-1"
}