FROM golang:1.21-alpine AS builder

RUN apk add --no-cache \
    git gcc g++ musl-dev linux-headers \
    pkgconf bash make \
    openssl-dev zlib-dev \
    lz4-dev zstd-dev \
    cyrus-sasl-dev

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build with musl tag
RUN CGO_ENABLED=1 GOOS=linux go build -tags musl -a -o sctp-interceptor cmd/interceptor/main.go

FROM alpine:3.18

RUN apk add --no-cache ca-certificates lz4-libs zstd-libs

COPY --from=builder /app/sctp-interceptor /usr/local/bin/

EXPOSE 36422/sctp 9090/tcp

ENTRYPOINT ["sctp-interceptor"]