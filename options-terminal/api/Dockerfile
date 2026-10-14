# Build context: options-terminal/ (needs contracts/ for the schema endpoint).
FROM rust:1-bookworm AS build
WORKDIR /src/api
ENV SQLX_OFFLINE=true
COPY api/Cargo.toml api/Cargo.lock ./
COPY api/.sqlx ./.sqlx
COPY api/src ./src
COPY api/tests ./tests
RUN cargo build --release --locked && cp target/release/options-terminal-api /options-terminal-api

FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="options-terminal-api" \
      com.options-terminal.project="options-terminal"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* && useradd --system --uid 10001 api
COPY --from=build /options-terminal-api /usr/local/bin/options-terminal-api
COPY contracts /app/contracts
USER api
EXPOSE 8080
CMD ["options-terminal-api"]
