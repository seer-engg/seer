-- Premium Pricing Launch Migration
-- Add payment tracking and early adopter fields to billing_profiles

-- Add new columns
ALTER TABLE billing_profiles ADD COLUMN IF NOT EXISTS payment_method_on_file BOOLEAN DEFAULT FALSE;
ALTER TABLE billing_profiles ADD COLUMN IF NOT EXISTS is_early_adopter BOOLEAN DEFAULT FALSE;
ALTER TABLE billing_profiles ADD COLUMN IF NOT EXISTS early_adopter_number INTEGER;

-- Grandfather existing users (created before Feb 1, 2026)
UPDATE billing_profiles
SET payment_method_on_file = TRUE
WHERE created_at < '2026-02-01 00:00:00+00'::TIMESTAMPTZ;

-- Create index for early adopter queries
CREATE INDEX IF NOT EXISTS idx_billing_profiles_early_adopter ON billing_profiles(is_early_adopter) WHERE is_early_adopter = TRUE;
