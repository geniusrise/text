![banner](./assets/banner.jpg)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

# Huggingface Bolts

This is a collection of generic streaming and (micro) batch bolts interfacing with the huggingface ecosystem.

**Table of Contents**

- [Huggingface Bolts](#huggingface-bolts)
  - [Usage](#usage)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Includes:

| No. | Name | Description | Output Type | Input Type |
| --- | ---- | ----------- | ----------- | ---------- |

## Usage

To test, first bring up all related services via the supplied docker-compose:

```bash
docker compose up -d
docker compose logs -f
```

These management consoles will be available:

| Console        | Link                    |
| -------------- | ----------------------- |
| Kafka UI       | http://localhost:8088/  |
| RabbitMQ UI    | http://localhost:15672/ |
| Localstack API | http://localhost:4566   |

Postgres can be accessed with:

```bash
docker exec -it geniusrise-postgres-1 psql -U postgres
```
