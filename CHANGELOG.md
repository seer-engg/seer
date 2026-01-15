# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features in development

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

## [0.1.4] - 2026-01-15

### Added
- LLM usage gates to manage credit consumption
- Google Analytics 4 integration for docs.getseer.dev with custom event tracking
- Anniversary billing cycle support
- Automatic frontend opening in development mode

### Fixed
- OAuth token exchange error handling improvements
- Intermittent OAuth callback failures with multiple workers
- Stripe customer creation race condition
- Usage tracking bugs
- LLM credit exhaustion capture and exception handling
- Supabase callback URL exemption

### Changed
- Increased timeout limits for long-running workflows
- Read limits from .env for easier override
- Removed Ultra pricing tier

### Removed
- Unnecessary local configuration files

## [0.0.9] - 2026-01-09

Initial public release with core workflow automation features.
