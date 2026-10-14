# Build context: options-terminal/. Static build served by unprivileged nginx, which proxies /v1
# to the API so the app is same-origin (VITE_API_BASE_URL empty → relative paths).
FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
LABEL org.opencontainers.image.title="options-terminal-frontend" \
      com.options-terminal.project="options-terminal"
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
