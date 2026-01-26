#!/usr/bin/env bash
# AWS Parameter Store List/View Script
# This script helps you view configuration values stored in AWS Parameter Store
#
# Usage:
#   ./scripts/view_aws_parameters.sh <environment>
#
# Example:
#   ./scripts/view_aws_parameters.sh dev
#   ./scripts/view_aws_parameters.sh prod

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Environment argument required${NC}"
    echo "Usage: $0 <environment>"
    echo "Example: $0 dev"
    echo "         $0 prod"
    exit 1
fi

ENVIRONMENT=$1
AWS_REGION=${AWS_REGION:-us-east-1}

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}  AWS Parameter Store for Environment: $ENVIRONMENT${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo ""
echo "Region: $AWS_REGION"
echo "Path: /${ENVIRONMENT}/"
echo ""

# Check if AWS CLI is configured
caller_identity=$(aws sts get-caller-identity --region "$AWS_REGION" 2>&1)
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: AWS CLI is not configured or you don't have permissions.${NC}"
    echo "Please run 'aws configure' first."
    exit 1
fi

echo -e "${GREEN}AWS credentials verified ✓${NC}"
echo "$caller_identity"
echo ""

# Function to mask secret values
mask_secret() {
    local value=$1
    local length=${#value}

    if [ $length -le 4 ]; then
        echo "***"
    else
        echo "***${value: -4}"
    fi
}

# Get all parameters
echo "Fetching parameters..."
parameters=$(aws ssm get-parameters-by-path \
    --path "/${ENVIRONMENT}/" \
    --recursive \
    --with-decryption \
    --region "$AWS_REGION" \
    --output json 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to fetch parameters${NC}"
    exit 1
fi

# Count parameters
param_count=$(echo "$parameters" | jq '.Parameters | length')

if [ "$param_count" -eq 0 ]; then
    echo -e "${YELLOW}No parameters found for environment '${ENVIRONMENT}'${NC}"
    echo ""
    echo "To add parameters, run:"
    echo "  ./scripts/setup_aws_parameters.sh $ENVIRONMENT"
    exit 0
fi

echo -e "${GREEN}Found $param_count parameter(s)${NC}"
echo ""

# Display parameters in a formatted table
echo -e "${BLUE}==============================================================${NC}"
printf "%-40s %-15s %s\n" "Parameter Name" "Type" "Value"
echo -e "${BLUE}==============================================================${NC}"

echo "$parameters" | jq -r '.Parameters[] | "\(.Name)|\(.Type)|\(.Value)"' | while IFS='|' read -r name type value; do
    # Extract parameter name without environment prefix
    param_name=$(echo "$name" | sed "s|^/${ENVIRONMENT}/||")

    # Mask SecureString values for security
    if [ "$type" = "SecureString" ]; then
        display_value=$(mask_secret "$value")
        printf "%-40s %-15s ${YELLOW}%s${NC}\n" "$param_name" "$type" "$display_value"
    else
        printf "%-40s %-15s %s\n" "$param_name" "$type" "$value"
    fi
done

echo -e "${BLUE}==============================================================${NC}"
echo ""

# Show legend
echo -e "${YELLOW}Note: SecureString values are masked for security (showing last 4 chars)${NC}"
echo ""

# Option to show full values
read -p "Show full values for SecureString parameters? (y/n): " show_full

if [ "$show_full" = "y" ]; then
    echo ""
    echo -e "${RED}WARNING: Displaying sensitive values!${NC}"
    echo -e "${BLUE}==============================================================${NC}"
    printf "%-40s %s\n" "Parameter Name" "Full Value"
    echo -e "${BLUE}==============================================================${NC}"

    echo "$parameters" | jq -r '.Parameters[] | select(.Type=="SecureString") | "\(.Name)|\(.Value)"' | while IFS='|' read -r name value; do
        param_name=$(echo "$name" | sed "s|^/${ENVIRONMENT}/||")
        printf "%-40s ${RED}%s${NC}\n" "$param_name" "$value"
    done

    echo -e "${BLUE}==============================================================${NC}"
    echo ""
fi

# Show total summary
echo "Summary:"
secure_count=$(echo "$parameters" | jq '[.Parameters[] | select(.Type=="SecureString")] | length')
string_count=$(echo "$parameters" | jq '[.Parameters[] | select(.Type=="String")] | length')

echo "  Total parameters: $param_count"
echo "  SecureString: $secure_count"
echo "  String: $string_count"
echo ""

# Show commands for management
echo "Useful commands:"
echo "  Delete parameter:"
echo "    aws ssm delete-parameter --name '/${ENVIRONMENT}/parameter_name' --region $AWS_REGION"
echo ""
echo "  Update parameter:"
echo "    aws ssm put-parameter --name '/${ENVIRONMENT}/parameter_name' --value 'new_value' --overwrite --region $AWS_REGION"
echo ""
echo "  Test configuration loading:"
echo "    export ENV=$ENVIRONMENT AWS_REGION=$AWS_REGION"
echo "    uv run python examples/config_priority_demo.py"
echo ""
