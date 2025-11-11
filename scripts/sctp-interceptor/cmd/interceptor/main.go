package main

import (
    "context"
    "encoding/hex"
    "flag"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"

    "github.com/confluentinc/confluent-kafka-go/v2/kafka"
    "github.com/ishidawataru/sctp"
)

type Config struct {
    ListenAddr      string
    UpstreamAddr    string
    KafkaBrokers    string
    KafkaTopic      string
    MetricsPort     int
    EnableKafka     bool
    EnableForward   bool
}

type Interceptor struct {
    config        *Config
    kafkaProducer *kafka.Producer
    metrics       *Metrics
}

type Metrics struct {
    mu               sync.RWMutex
    connections      uint64
    messagesReceived uint64
    bytesReceived    uint64
    messagesForwarded uint64
    messagesKafka    uint64
    errors           uint64
}

func main() {
    config := &Config{}
    flag.StringVar(&config.ListenAddr, "listen", ":36422", "SCTP listen address")
    flag.StringVar(&config.UpstreamAddr, "upstream", "sctp-service.ricplt:36422", "Upstream E2 termination address")
    flag.StringVar(&config.KafkaBrokers, "kafka", "my-kafka.default.svc.cluster.local:9092", "Kafka broker addresses")
    flag.StringVar(&config.KafkaTopic, "topic", "e2-data", "Kafka topic for E2 messages")
    flag.IntVar(&config.MetricsPort, "metrics", 9090, "Metrics port")
    flag.BoolVar(&config.EnableKafka, "enable-kafka", true, "Enable Kafka streaming")
    flag.BoolVar(&config.EnableForward, "enable-forward", true, "Enable forwarding to upstream")
    flag.Parse()

    interceptor := &Interceptor{
        config:  config,
        metrics: &Metrics{},
    }

    // Initialize Kafka if enabled
    if config.EnableKafka {
        kafkaConfig := kafka.ConfigMap{
            "bootstrap.servers": config.KafkaBrokers,
            "client.id":        "sctp-interceptor",
        }
        
        producer, err := kafka.NewProducer(&kafkaConfig)
        if err != nil {
            log.Printf("Warning: Failed to create Kafka producer: %v", err)
            config.EnableKafka = false
        } else {
            interceptor.kafkaProducer = producer
            defer producer.Close()
            
            // Start event handler
            go func() {
                for e := range producer.Events() {
                    switch ev := e.(type) {
                    case *kafka.Message:
                        if ev.TopicPartition.Error != nil {
                            log.Printf("Kafka delivery failed: %v", ev.TopicPartition.Error)
                            interceptor.metrics.incrementErrors()
                        }
                    }
                }
            }()
        }
    }

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
    go func() {
        <-sigChan
        log.Println("Shutting down...")
        cancel()
    }()

    go interceptor.startMetricsServer()

    if err := interceptor.Start(ctx); err != nil {
        log.Fatalf("Failed to start: %v", err)
    }
}

func (i *Interceptor) Start(ctx context.Context) error {
    addr, err := sctp.ResolveSCTPAddr("sctp", i.config.ListenAddr)
    if err != nil {
        return fmt.Errorf("resolve addr: %w", err)
    }

    listener, err := sctp.ListenSCTP("sctp", addr)
    if err != nil {
        return fmt.Errorf("listen: %w", err)
    }
    defer listener.Close()

    log.Printf("SCTP interceptor listening on %s", i.config.ListenAddr)
    log.Printf("Forwarding enabled: %v, to: %s", i.config.EnableForward, i.config.UpstreamAddr)
    log.Printf("Kafka enabled: %v, topic: %s", i.config.EnableKafka, i.config.KafkaTopic)

    for {
        select {
        case <-ctx.Done():
            return nil
        default:
            conn, err := listener.AcceptSCTP()
            if err != nil {
                log.Printf("Accept error: %v", err)
                continue
            }
            go i.handleConnection(ctx, conn)
        }
    }
}

func (i *Interceptor) handleConnection(ctx context.Context, clientConn *sctp.SCTPConn) {
    defer clientConn.Close()
    i.metrics.incrementConnections()
    
    clientAddr := clientConn.RemoteAddr()
    log.Printf("New connection from %s", clientAddr)
    
    // Connect to upstream if forwarding is enabled
    var upstreamConn *sctp.SCTPConn
    if i.config.EnableForward {
        upstreamAddr, err := sctp.ResolveSCTPAddr("sctp", i.config.UpstreamAddr)
        if err != nil {
            log.Printf("Failed to resolve upstream address: %v", err)
            i.metrics.incrementErrors()
            return
        }
        
        upstreamConn, err = sctp.DialSCTP("sctp", nil, upstreamAddr)
        if err != nil {
            log.Printf("Failed to connect to upstream %s: %v", i.config.UpstreamAddr, err)
            i.metrics.incrementErrors()
            return
        }
        defer upstreamConn.Close()
        log.Printf("Connected to upstream %s", i.config.UpstreamAddr)
        
        // Start upstream to client forwarding
        go i.forwardData(ctx, upstreamConn, clientConn, "upstream->client")
    }
    
    // Forward client to upstream
    i.forwardData(ctx, clientConn, upstreamConn, "client->upstream")
    
    log.Printf("Connection closed from %s", clientAddr)
}

func (i *Interceptor) forwardData(ctx context.Context, from *sctp.SCTPConn, to *sctp.SCTPConn, direction string) {
    buffer := make([]byte, 65536)
    
    for {
        select {
        case <-ctx.Done():
            return
        default:
            n, info, err := from.SCTPRead(buffer)
            if err != nil {
                if err.Error() != "EOF" {
                    log.Printf("Read error (%s): %v", direction, err)
                    i.metrics.incrementErrors()
                }
                return
            }
            
            if n > 0 {
                i.metrics.recordMessage(n)
                
                // Log message details
                if info != nil {
                    log.Printf("[%s] %d bytes, Stream: %d, PPID: %d", direction, n, info.Stream, info.PPID)
                } else {
                    log.Printf("[%s] %d bytes (no SCTP info)", direction, n)
                }
                
                // Log first 64 bytes for debugging
                if n > 64 {
                    log.Printf("[%s] First 64 bytes: %s", direction, hex.EncodeToString(buffer[:64]))
                } else {
                    log.Printf("[%s] Data: %s", direction, hex.EncodeToString(buffer[:n]))
                }
                
                // Forward to upstream if enabled and this is client->upstream
                if to != nil && i.config.EnableForward {
                    _, err = to.SCTPWrite(buffer[:n], info)
                    if err != nil {
                        log.Printf("Write error (%s): %v", direction, err)
                        i.metrics.incrementErrors()
                        return
                    }
                    i.metrics.incrementForwarded()
                }
                
                // Send to Kafka if enabled and this is client->upstream
                if i.config.EnableKafka && direction == "client->upstream" && i.kafkaProducer != nil {
                    i.sendToKafka(buffer[:n], info)
                }
            }
        }
    }
}

func (i *Interceptor) sendToKafka(data []byte, info *sctp.SndRcvInfo) {
    // Create a copy of the data
    dataCopy := make([]byte, len(data))
    copy(dataCopy, data)
    
    // Create headers
    headers := []kafka.Header{
        {Key: "timestamp", Value: []byte(fmt.Sprintf("%d", time.Now().UnixNano()))},
        {Key: "direction", Value: []byte("client->upstream")},
    }
    
    if info != nil {
        headers = append(headers,
            kafka.Header{Key: "sctp_stream", Value: []byte(fmt.Sprintf("%d", info.Stream))},
            kafka.Header{Key: "sctp_ppid", Value: []byte(fmt.Sprintf("%d", info.PPID))},
        )
    }
    
    // Identify message type from data
    if len(data) > 2 {
        if data[0] == 0x00 && data[1] == 0x01 {
            headers = append(headers, kafka.Header{Key: "message_type", Value: []byte("E2AP_INITIATING")})
        } else if data[0] == 0x20 && data[1] == 0x01 {
            headers = append(headers, kafka.Header{Key: "message_type", Value: []byte("E2AP_SUCCESSFUL")})
        } else if data[0] == 0x40 && data[1] == 0x01 {
            headers = append(headers, kafka.Header{Key: "message_type", Value: []byte("E2AP_UNSUCCESSFUL")})
        }
    }
    
    // Create Kafka message
    message := &kafka.Message{
        TopicPartition: kafka.TopicPartition{
            Topic:     &i.config.KafkaTopic,
            Partition: kafka.PartitionAny,
        },
        Value: dataCopy,
        Headers: headers,
    }
    
    // Send to Kafka
    err := i.kafkaProducer.Produce(message, nil)
    if err != nil {
        log.Printf("Failed to produce to Kafka: %v", err)
        i.metrics.incrementErrors()
    } else {
        i.metrics.incrementKafka()
        log.Printf("Sent %d bytes to Kafka topic %s", len(data), i.config.KafkaTopic)
    }
}

func (i *Interceptor) startMetricsServer() {
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
        i.metrics.mu.RLock()
        defer i.metrics.mu.RUnlock()
        fmt.Fprintf(w, "# HELP sctp_interceptor_connections_total Total number of connections\n")
        fmt.Fprintf(w, "sctp_interceptor_connections_total %d\n", i.metrics.connections)
        fmt.Fprintf(w, "# HELP sctp_interceptor_messages_total Total number of messages received\n")
        fmt.Fprintf(w, "sctp_interceptor_messages_total %d\n", i.metrics.messagesReceived)
        fmt.Fprintf(w, "# HELP sctp_interceptor_bytes_total Total bytes received\n")
        fmt.Fprintf(w, "sctp_interceptor_bytes_total %d\n", i.metrics.bytesReceived)
        fmt.Fprintf(w, "# HELP sctp_interceptor_messages_forwarded_total Messages forwarded to upstream\n")
        fmt.Fprintf(w, "sctp_interceptor_messages_forwarded_total %d\n", i.metrics.messagesForwarded)
        fmt.Fprintf(w, "# HELP sctp_interceptor_messages_kafka_total Messages sent to Kafka\n")
        fmt.Fprintf(w, "sctp_interceptor_messages_kafka_total %d\n", i.metrics.messagesKafka)
        fmt.Fprintf(w, "# HELP sctp_interceptor_errors_total Total errors\n")
        fmt.Fprintf(w, "sctp_interceptor_errors_total %d\n", i.metrics.errors)
    })
    
    log.Printf("Metrics server on :%d", i.config.MetricsPort)
    http.ListenAndServe(fmt.Sprintf(":%d", i.config.MetricsPort), nil)
}

func (m *Metrics) incrementConnections() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.connections++
}

func (m *Metrics) recordMessage(bytes int) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.messagesReceived++
    m.bytesReceived += uint64(bytes)
}

func (m *Metrics) incrementForwarded() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.messagesForwarded++
}

func (m *Metrics) incrementKafka() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.messagesKafka++
}

func (m *Metrics) incrementErrors() {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.errors++
}