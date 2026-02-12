import { Header } from "@/components/header";
import { VideoUploadSection } from "@/components/sections/video-upload-section";
import { UploadGallerySection } from "@/components/sections/upload-gallery-section";
import { FooterSection } from "@/components/sections/footer-section";
import { HeroSection } from "@/components/sections/hero-section";

export default function UploadPage() {
  return (
    <main className="min-h-screen bg-background">
      <Header />
      {/* <HeroSection /> */}
      <VideoUploadSection />
      <UploadGallerySection />
      <FooterSection />
    </main>
  );
}
