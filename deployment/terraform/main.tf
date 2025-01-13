terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
  credentials = file(var.credentials)
  project     = var.project
  region      = var.region
}

data "google_project" "project" {
}

# Create serverless VPC connector
resource "google_vpc_access_connector" "connector" {
  name          = "cloudrun-vpc-connector"
  ip_cidr_range = "10.8.0.0/28"
  network       = "default"
  machine_type  = "f1-micro"
  min_instances = 2
  max_instances = 3
  region        = var.region
}

resource "google_cloud_run_service" "predict-api" {
  name     = "predict-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project}/elo-prediction-api:latest"
        ports {
          container_port = 8000
        }
      }
    }
    #Metadata for Revision
     metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-egress"    = "all"
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  #Metadata for Service
  metadata {
    annotations = {
      "run.googleapis.com/ingress" = "internal"
    }
  }

  depends_on = [
    google_vpc_access_connector.connector
  ]
}

resource "google_cloud_run_service_iam_binding" "predict-api-invoker" {
  location = google_cloud_run_service.predict-api.location
  service  = google_cloud_run_service.predict-api.name
  role     = "roles/run.invoker"
  members = [
    "allUsers"
  ]
}


resource "google_cloud_run_service" "streamlit-frontend" {
  name       = "predict-chess-elo-streamlit-frontend"
  depends_on = [google_cloud_run_service.predict-api]
  location   = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project}/predict-chess-elo-streamlit-frontend:latest"
        env {
          name  = "PREDICTION_SERVICE_URL"
          value = "${google_cloud_run_service.predict-api.status[0].url}"
        }
        ports {
          container_port = 8501
        }
      }
    }

    #Metadata for Revision
     metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-egress"    = "all"
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  #Metadata for Service
  metadata {
    annotations = {
      "run.googleapis.com/ingress" = "all"
    }
  }

}

resource "google_cloud_run_service_iam_binding" "streamlit-frontend-invoker" {
  location = google_cloud_run_service.streamlit-frontend.location
  service  = google_cloud_run_service.streamlit-frontend.name
  role     = "roles/run.invoker"
  members = [
    "allUsers"
  ]
}

output "internal_predict_api_url" {
  value = google_cloud_run_service.predict-api.status[0].url
}

output "frontend_url" {
  value = google_cloud_run_service.streamlit-frontend.status[0].url
}