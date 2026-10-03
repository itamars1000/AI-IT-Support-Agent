import unittest

from fastapi.testclient import TestClient

from app.main import app


class ChatPageTests(unittest.TestCase):
    def test_chat_page_and_its_assets_are_served_without_provider_keys(self):
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        for path, content_type in (("/assets/styles.css", "text/css"),
                                   ("/assets/chat.js", "javascript"),
                                   ("/assets/favicon.svg", "image/svg+xml")):
            asset = client.get(path)
            self.assertEqual(asset.status_code, 200)
            self.assertIn(content_type, asset.headers["content-type"])

    def test_private_files_are_not_exposed_by_asset_routes(self):
        client = TestClient(app)
        for path in ("/assets/.env", "/assets/../config.py", "/assets/%2e%2e/config.py"):
            self.assertEqual(client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
