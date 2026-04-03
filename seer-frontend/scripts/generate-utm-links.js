#!/usr/bin/env node

/**
 * UTM Link Generator
 *
 * Generates UTM-enriched links for different platforms
 * Usage: npm run utm [campaign_name]
 * Example: npm run utm demo_video
 */

// Get campaign name from command line args or use default
const args = process.argv.slice(2);
const campaignName = args[0] || 'demo_video';

// Base URL
const baseUrl = 'https://app.getseer.dev';

// Platform configurations
const platforms = [
  {
    name: 'HackerNews',
    utm_source: 'hackernews',
    utm_medium: 'community',
    variants: [
      { label: 'post', utm_content: 'post' }
    ]
  },
  {
    name: 'YouTube',
    utm_source: 'youtube',
    utm_medium: 'video',
    variants: [
      { label: 'description', utm_content: 'description' },
      { label: 'pinned comment', utm_content: 'pinned_comment' }
    ]
  },
  {
    name: 'LinkedIn',
    utm_source: 'linkedin',
    utm_medium: 'social',
    variants: [
      { label: 'post', utm_content: 'post' }
    ]
  },
  {
    name: 'Reddit',
    utm_source: 'reddit',
    utm_medium: 'community',
    variants: [
      { label: 'post', utm_content: 'post' },
      { label: 'comment', utm_content: 'comment' }
    ]
  },
  {
    name: 'Twitter/X',
    utm_source: 'twitter',
    utm_medium: 'social',
    variants: [
      { label: 'post', utm_content: 'post' }
    ]
  }
];

/**
 * Generate UTM link
 */
function generateLink(source, medium, campaign, content) {
  const params = new URLSearchParams({
    utm_source: source,
    utm_medium: medium,
    utm_campaign: campaign,
    utm_content: content
  });

  return `${baseUrl}?${params.toString()}`;
}

/**
 * Print formatted output
 */
function printLinks() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 UTM Link Generator for Seer');
  console.log('='.repeat(60));
  console.log(`\nCampaign: ${campaignName}\n`);

  platforms.forEach(platform => {
    console.log(`\n${platform.name}:`);
    console.log('-'.repeat(60));

    platform.variants.forEach(variant => {
      const link = generateLink(
        platform.utm_source,
        platform.utm_medium,
        campaignName,
        variant.utm_content
      );

      if (platform.variants.length > 1) {
        console.log(`  ${variant.label}:`);
        console.log(`  ${link}`);
      } else {
        console.log(`  ${link}`);
      }
    });
  });

  console.log('\n' + '='.repeat(60));
  console.log('✅ Links generated successfully!');
  console.log('💡 Copy and paste these links into your marketing materials');
  console.log('='.repeat(60) + '\n');
}

// Run the script
printLinks();
