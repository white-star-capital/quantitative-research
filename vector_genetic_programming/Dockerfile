# Dashboard image for Vector Genetic Programming.
#
# Shows the committed run in results/ as a web page. Deliberately NOT the image
# that runs the experiment: scripts/build_dashboard.py is pure standard library,
# so this needs no vectorbt, no numba and no compiler, and builds in seconds
# instead of minutes. `make vgp` runs the experiment itself.
#
#   make start        build this image and serve the dashboard
#   docker build -t vgp-dashboard .
#   docker run --rm -p 8000:8000 vgp-dashboard
#
# Pinned to a digest-free but explicit minor version: python:3.12-slim tracks
# security patches within 3.12, which is what the project targets. The page is
# static once built, so a patch release cannot change what it shows.
FROM python:3.12-slim

# Run as a non-root user. Nothing here needs privileges, and the container
# serves a port.
RUN useradd --create-home --uid 10001 vgp
WORKDIR /app

# Only what the dashboard build reads. Copying results/ and the template
# separately from the rest of the tree keeps the layer cache useful: editing
# project source does not rebuild this image, editing results/ does.
COPY --chown=vgp:vgp dashboard/template.html dashboard/template.html
COPY --chown=vgp:vgp scripts/build_dashboard.py scripts/build_dashboard.py
COPY --chown=vgp:vgp results/ results/

USER vgp

# Build the page at image build time, so a broken or stale results/ fails the
# build rather than producing a container that serves nothing.
RUN python scripts/build_dashboard.py && test -s dashboard/index.html

EXPOSE 8000

# http.server is a local viewer, not a production server: it is single-threaded
# and has no TLS. That is the right tool here — the page is static, read-only,
# and meant to be opened on the machine that ran the experiment.
CMD ["python", "-m", "http.server", "8000", "--directory", "dashboard", "--bind", "0.0.0.0"]
