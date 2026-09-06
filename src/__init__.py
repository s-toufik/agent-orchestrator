import os
from importlib.metadata import PackageMetadata, metadata

import dotenv

dotenv.load_dotenv()

APPLICATION_NAME: str = "agent"
project_metadata: PackageMetadata = metadata(APPLICATION_NAME)
APPLICATION_VERSION: str = project_metadata.get("version", "X.X.X")
APPLICATION_API_ROOT_PATH: str = os.getenv("APPLICATION_API_ROOT_PATH", f"/{APPLICATION_NAME}")
APPLICATION_DEPLOYMENT_ENVIRONMENT: str = os.getenv("DEPLOYMENT_ENVIRONMENT", "unknown")
APPLICATION_AUTHORS_EMAIL: str = project_metadata.get("Author-email", "N/A")
