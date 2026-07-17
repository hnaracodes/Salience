import { NavBar } from '@/components/NavBar'

import { HeroSection } from '@/components/HeroSection'

import { UseCaseSection } from '@/components/marketing/UseCaseSection'

import { ScrollProductJourney } from '@/components/marketing/ScrollProductJourney'

import { HowItWorksSection } from '@/components/HowItWorksSection'

import { FeaturesSection } from '@/components/FeaturesSection'

import { ViewerFinaleSection } from '@/components/marketing/ViewerFinaleSection'

import { CTASection } from '@/components/CTASection'

import { SiteFooter } from '@/components/SiteFooter'

import { ScrollScene } from '@/components/ScrollScene'



export default function HomePage() {

  return (

    <>

      <NavBar />

      <main className="bg-canvas">

        <HeroSection />

        <UseCaseSection />

        <ScrollProductJourney />

        <HowItWorksSection />

        <FeaturesSection />

        <ViewerFinaleSection />

        <CTASection />

      </main>

      <SiteFooter />

      <ScrollScene />

    </>

  )

}

