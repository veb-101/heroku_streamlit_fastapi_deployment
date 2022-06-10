# heroku_deployment_streamlit_fastapi

We are deploying Streamlit and FastAPI for TensorFlow image classification on Heroku.

To deploy multiple apps from a single repo, do the following steps:

1. In this project, we will use Streamlit for the frontend and FastAPI for the backend. Create two apps on Heroku.
2. For each app, add the language buildpack (python in this case) and [subdir-heroku-buildpack](https://github.com/timanovsky/subdir-heroku-buildpack) buildpack.
3. Buildpack arrangement
   1. subdir-Heroku-buildpack
   2. language buildpack
4. Set PROJECT_PATH as key and the folder name/path as value. This is required by subdir-heuroku-buildpack.
5. Connect the GitHub repo for both apps. In this project, we deployed the backend (FastAPI) app first, and the app's public URL was used in the frontend (Streamlit) app.
