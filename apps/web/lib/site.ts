/** Site-wide constants for marketing, footer, and legal pages. */
export const SITE = {
  /** One-word company / product brand */
  name: 'Salience',
  tagline: 'Neural UX analysis',
  headline: 'finds which parts of your site hold attention — before users bounce.',
  description:
    'Paste any public URL. Salience crawls up to 8 same-origin pages, records walkthrough video, runs TRIBE v2 cortical modeling, and returns heatmaps, copy scores, and per-element ratings in an interactive viewer.',
  legalEntity: 'Salience, Inc.',
  domain: 'salience.app',
  contactEmail: 'legal@salience.app',
  supportEmail: 'support@salience.app',
  privacyEmail: 'privacy@salience.app',
  securityEmail: 'security@salience.app',
  lastUpdated: 'June 17, 2026',
  address: 'Delaware, United States',
  /** Example URL shown in product mocks */
  exampleUrl: 'acme.com',
  founder: {
    name: 'Sai Hruday Reddy Nara',
    title: 'Founder & CEO',
    location: 'Los Gatos, California',
    bio: 'Builder, hackathon organizer, and ML researcher. Founded CodeCraft Academy to expand CS education, led LG Hacks, and ships products from fitness apps to neural UX tooling.',
    links: {
      linkedin: 'https://www.linkedin.com/in/sai-hruday-reddy-nara-5b03632b3/',
      github: 'https://github.com/EpicCoder1234',
      devpost: 'https://devpost.com/EpicCoder1234',
    },
  },
} as const

export const NAV_LINKS = [
  { href: '/#use-cases', label: 'Use cases' },
  { href: '/#product-journey', label: 'Pipeline' },
  { href: '/#how-it-works', label: 'How it works' },
  { href: '/about', label: 'About' },
  { href: '/security', label: 'Security' },
] as const

/** Muted partner-style logos for hero social proof strip */
export const SOCIAL_PROOF = [
  'Threadmind',
  'VitaHealth',
  'BoxMedia',
  'NovaTech',
] as const
