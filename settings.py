import os
import dotenv

dotenv.load_dotenv(".env")


class Settings:
    data_path: str = os.getenv("DATA_PATH", None)
    separator: str = ','

    def get_settings(self):
        return {"path": self.data_path, "sep": self.separator}
