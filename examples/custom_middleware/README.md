# AUA Custom Middleware Example

Demonstrates how to write and register custom middleware.
This example includes a PII redaction middleware and an audit middleware.

## Registration in aua_config.yaml

```yaml
middleware:
  - import_path: aua.middleware:PIIRedactionMiddleware
    config:
      patterns:
        - "\\d{3}-\\d{2}-\\d{4}"   # SSN
  - import_path: aua.middleware:AuditMiddleware
```

## Testing

```bash
aua extensions test \
  --kind middleware \
  --import-path aua.middleware:PIIRedactionMiddleware
```

All built-in middleware is available via `aua.middleware`. Custom middleware
goes in your project's `plugins/` directory.
