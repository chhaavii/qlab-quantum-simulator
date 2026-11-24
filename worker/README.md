# Hosted API

`index.js` is a JavaScript port of `web/app.py`: it answers the same five
`/api/*` endpoints for the deployed site, so the page in `web/static` runs
without a Python server.

The deployment is described by `sites.deploy.json` at the repository root,
which pairs this Worker with `web/static` as its static assets. The live site
is <https://qlab-quantum-simulator-l4e9zj.freebuff.site>.
