"use client";

import Image from "next/image";
import { useState } from "react";
import { ChevronRight } from "lucide-react";

export function UploadGallerySection() {
  const [activeTab, setActiveTab] = useState<"images" | "templates">("images");

  const images = [
    { id: 1, src: "/images/bottle-bike.png", title: "Adventure Awaits" },
    { id: 2, src: "/images/bottle-lake.png", title: "Serene Moments" },
    { id: 3, src: "/images/bottle-water.png", title: "Crystal Clear" },
    { id: 4, src: "/images/bottle-stream.png", title: "Nature's Flow" },
    { id: 5, src: "/images/bottle-fire.png", title: "Warmth & Comfort" },
    { id: 6, src: "/images/bottle-snow.png", title: "Winter Magic" },
    { id: 7, src: "/images/bottle-mountain.png", title: "Peak Experience" },
    { id: 8, src: "/images/bottle-canyon.png", title: "Endless Horizons" },
  ];

  const templates = [
    { id: 1, src: "/images/led-flashlight-bottle.png", title: "LED Flashlight", model: "3D" },
    { id: 2, src: "/images/accessory-strap.png", title: "Carrying Strap", model: "3D" },
    { id: 3, src: "/images/accessory-charger.png", title: "Quick Charger", model: "3D" },
    { id: 4, src: "/images/accessory-sleeve.png", title: "Protective Sleeve", model: "3D" },
    { id: 5, src: "/images/accessory-bike-mount.png", title: "Bike Mount", model: "3D" },
    { id: 6, src: "/images/accessory-carabiner.png", title: "Carabiner", model: "3D" },
    { id: 7, src: "/images/accessory-speaker-base.png", title: "Speaker Base", model: "3D" },
    { id: 8, src: "/images/heating-campfire.png", title: "Heat Element", model: "3D" },
  ];

  const items = activeTab === "images" ? images : templates;

  return (
    <section className="bg-secondary/30 px-4 py-20 md:py-32">
      <div className="mx-auto max-w-7xl">
        {/* Section Header */}
        <div className="mb-12 text-center">
          <h2 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Explore Our Collection
          </h2>
          <p className="mb-8 text-lg text-muted-foreground">
            Discover premium products and interactive 3D templates
          </p>

          {/* Tabs */}
          <div className="flex justify-center gap-4">
            <button
              onClick={() => setActiveTab("images")}
              className={`rounded-full px-6 py-2 font-semibold transition-all ${
                activeTab === "images"
                  ? "bg-foreground text-background"
                  : "border-2 border-border text-foreground hover:border-foreground"
              }`}
            >
              Product Images
            </button>
            <button
              onClick={() => setActiveTab("templates")}
              className={`rounded-full px-6 py-2 font-semibold transition-all ${
                activeTab === "templates"
                  ? "bg-foreground text-background"
                  : "border-2 border-border text-foreground hover:border-foreground"
              }`}
            >
              3D Templates
            </button>
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {items.map((item) => (
            <div
              key={item.id}
              className="group relative overflow-hidden rounded-xl border border-border bg-card transition-all duration-300 hover:border-foreground hover:shadow-lg"
            >
              {/* Image */}
              <div className="relative aspect-square w-full overflow-hidden bg-secondary">
                <Image
                  src={item.src || "/placeholder.svg"}
                  alt={item.title}
                  fill
                  className="object-cover transition-transform duration-300 group-hover:scale-105"
                />
                
                {/* Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              </div>

              {/* Content */}
              <div className="p-4">
                <h3 className="font-semibold text-foreground">{item.title}</h3>
                {activeTab === "templates" && (
                  <div className="mt-2 inline-block rounded-full bg-secondary/50 px-3 py-1 text-xs text-muted-foreground">
                    {(item as any).model}
                  </div>
                )}
              </div>

              {/* Hover Button */}
              <button className="absolute bottom-4 right-4 translate-y-2 rounded-full bg-foreground p-2 text-background opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100">
                <ChevronRight size={20} />
              </button>
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className="mt-16 text-center">
          <p className="mb-6 text-muted-foreground">
            Interested in our products? Explore the full catalog or request a demo.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="rounded-full bg-foreground px-8 py-3 font-semibold text-background transition-all hover:opacity-90 active:scale-95">
              View All Products
            </button>
            <button className="rounded-full border-2 border-foreground px-8 py-3 font-semibold text-foreground transition-all hover:bg-foreground hover:text-background active:scale-95">
              Request Demo
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
