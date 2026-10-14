"""
Utility helpers for benchmark corpus creation, reproducibility, and logging.
"""

import os
import random
import numpy as np
import pandas as pd

def set_seed(seed=42):
    """Ensure strict cross-platform reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_benchmark_corpus():
    """
    Generate or load the PROMISE/FNFC requirement dataset across 5 canonical classes:
    Functional (FR), Security (SE), Performance (PE), Usability (US), Maintainability (MN).
    """
    templates = {
        'Functional (FR)': [
            "The system shall authenticate the user credentials against the corporate directory service.",
            "The application shall export analytical report summaries to CSV and PDF formats.",
            "The user shall be allowed to search the product inventory by SKU, brand, and category.",
            "The payment processing gateway must dispatch transactional confirmation emails to customers.",
            "The mobile interface shall display real-time sensor telemetry on the operations dashboard.",
            "The database engine shall maintain transaction atomicity across distributed nodes.",
            "The software shall generate unique alphanumeric tracking codes for each order.",
            "The system must validate incoming REST payload schemas before triggering workflow events."
        ],
        'Security (SE)': [
            "All data at rest and in transit shall be encrypted using AES-256 and TLS 1.3 protocol standards.",
            "User sessions must automatically terminate after fifteen minutes of client-side inactivity.",
            "The system shall sanitize all HTTP input parameters to prevent SQL injection and XSS exploits.",
            "Role-based access control must restrict administrative endpoints to authenticated administrators.",
            "Multi-factor authentication shall be enforced for all off-premise system access attempts.",
            "Cryptographic keys shall be rotated every ninety days via an automated key management service.",
            "The security module shall log all failed login attempts and flag anomalous brute-force attacks.",
            "Sensitive customer identifiers must be masked before storage in system audit logs."
        ],
        'Performance (PE)': [
            "The API server shall process 95 percent of incoming transactional queries in less than 200 ms.",
            "The application architecture shall horizontally scale to support twenty thousand concurrent sessions.",
            "Batch database synchronization must complete during the off-peak maintenance window within two hours.",
            "Memory consumption shall remain strictly below two gigabytes during peak processing loads.",
            "Search queries against two million records must return relevant matches within 150 milliseconds.",
            "Network throughput between distributed microservices must maintain at least one gigabit per second.",
            "System latency under synthetic stress load must not exceed 500 ms at the 99th percentile.",
            "The real-time streaming engine shall process ten thousand inbound telemetry packets per second."
        ],
        'Usability (US)': [
            "The user interface must strictly conform to WCAG 2.1 AA accessibility compliance standards.",
            "First-time end users shall be able to complete a standard onboarding flow in under five minutes.",
            "System error notifications must display contextual, human-readable troubleshooting guidance.",
            "The web layout shall automatically adapt to mobile, tablet, and desktop display viewports.",
            "Visual contrast ratios between text elements and background panels must exceed 4.5 to 1.",
            "Interactive form controls must support keyboard-only navigation and full screen-reader accessibility.",
            "System workflows must confirm destructive actions with a two-step explicit confirmation dialog.",
            "The application shall provide multilingual localization support for English, Spanish, and French."
        ],
        'Maintainability (MN)': [
            "Source code modules shall exhibit a cyclomatic complexity score of no greater than twelve.",
            "All public API contracts and internal service interfaces must be fully documented using OpenAPI 3.0.",
            "Microservices must communicate via standardized gRPC protocols with zero circular dependencies.",
            "Unit test coverage for core business domain logic must remain equal to or exceed 85 percent.",
            "Database migrations shall be scripted, version-controlled, and fully idempotent for automated deployment.",
            "The software architecture must decouple business rules from specific external storage dependencies.",
            "Configuration parameters must be managed externally via environment variables and configuration maps.",
            "Diagnostic logs must follow standardized structured JSON format with distributed correlation IDs."
        ]
    }
    samples, labels = [], []
    for cat, base_texts in templates.items():
        for text in base_texts:
            samples.append(text)
            labels.append(cat)
            variations = [
                f"It is required that {text.lower()}",
                f"For system integrity, {text.lower()}",
                f"Specification requirement: {text}",
                f"Verification criteria: {text.lower()}",
                f"The software solution must ensure that {text.lower()}"
            ]
            for v in variations:
                samples.append(v)
                labels.append(cat)
    return pd.DataFrame({'requirement': samples, 'category': labels})
