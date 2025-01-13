variable "credentials" {
  description = "My Credentials"
  default     = "<PATH TO YOUR SERVICE ACCOUNT CREDENTIALS FILE HERE>"
}

variable "project" {
  description = "Project"
  default     = "<YOUR GCP PROJECT NAME HERE>"
}

variable "region" {
  description = "Your project region"
  default     = "us-central1"
  type        = string
}

variable "zone" {
  description = "Your project zone"
  default     = "us-central1-a"
  type        = string
}

variable "location" {
  description = "Project Location"
  default     = "US"
}

variable "gcs_storage_class" {
  description = "Bucket Storage Class"
  default     = "STANDARD"
}