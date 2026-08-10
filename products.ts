import goldenMistPair from "@/assets/products/golden-mist-pair.jpg";
import milkDipCup from "@/assets/products/milk-dip-cup.jpg";
import harvestMoonCup from "@/assets/products/harvest-moon-cup.jpg";
import springBlade from "@/assets/products/spring-blade.jpg";
import classicSet from "@/assets/products/classic-set.jpg";
import countryFeastSet from "@/assets/products/country-feast-set.jpg";
import earthSkyPlanter from "@/assets/products/earth-sky-planter.jpg";
import goldenBlushCup from "@/assets/products/golden-blush-cup.jpg";
import saltSpout from "@/assets/products/salt-spout.jpg";

export interface Product {
  slug: string;
  name: string;
  price: number;
  originalPrice?: number;
  image: string;
  description: string;
  badge?: "sale" | "sold-out";
  availability?: string;
}

export const products: Product[] = [
  {
    slug: "spring-blade",
    name: "Moss Stitch Cardigan",
    price: 50,
    image: springBlade,
    description: "Hand-knitted from locally sourced wool using a textured moss stitch pattern. The relaxed silhouette drapes naturally, with hand-carved wooden buttons and ribbed cuffs for a refined finish.",
  },
  {
    slug: "classic-set",
    name: "Cable Knit Sweater",
    price: 50,
    image: classicSet,
    description: "A timeless cable knit pullover crafted from undyed heritage wool. Each twist and braid is worked by hand, creating a rich surface texture that softens beautifully with wear.",
  },
  {
    slug: "country-feast-set",
    name: "Merino Wool Scarf Set",
    price: 50,
    image: countryFeastSet,
    description: "A coordinated set of two generously sized scarves in complementary earth tones. Knitted from extra-fine merino for a soft hand feel, with hand-twisted fringe detailing.",
  },
  {
    slug: "earth-sky-planter",
    name: "Alpaca Blend Beanie",
    price: 40,
    originalPrice: 50,
    image: earthSkyPlanter,
    badge: "sale",
    description: "A cozy ribbed beanie knitted from a baby alpaca and wool blend. Lightweight yet warm, with a gently slouched crown and folded brim that fits all head sizes.",
  },
  {
    slug: "golden-blush-cup",
    name: "Ribbed Wool Vest",
    price: 50,
    image: goldenBlushCup,
    description: "A versatile layering piece in a deep rib knit, crafted from medium-weight lambswool. The V-neck and clean armholes make it perfect over a shirt or worn alone in warmer months.",
  },
  {
    slug: "harvest-moon-cup",
    name: "Chunky Knit Pullover",
    price: 50,
    image: harvestMoonCup,
    availability: "Only 4 available",
    description: "Our most substantial knit — a chunky-gauge pullover worked in a bold herringbone pattern. Made from hand-dyed wool in our signature rust colorway, with dropped shoulders and a relaxed fit.",
  },
  {
    slug: "milk-dip-cup",
    name: "Cashmere Wrap",
    price: 50,
    image: milkDipCup,
    description: "An oversized wrap knitted from pure Mongolian cashmere in a delicate stockinette stitch. Finished with a subtle fringe edge, this piece is as soft as it is elegant.",
  },
  {
    slug: "salt-spout",
    name: "Heritage Mittens",
    price: 50,
    image: saltSpout,
    badge: "sold-out",
    description: "Traditional stranded colourwork mittens inspired by Nordic knitting heritage. Knitted from sturdy Shetland wool in natural undyed shades for a piece that tells a story.",
  },
  {
    slug: "golden-mist-pair",
    name: "Lambswool Socks Pair",
    price: 50,
    image: goldenMistPair,
    description: "A pair of mid-calf socks knitted from brushed lambswool with reinforced heels and toes. The ribbed leg ensures a snug fit, while the soft fiber keeps feet warm all day.",
  },
];

export const featuredProducts = [
  products.find(p => p.slug === "golden-mist-pair")!,
  products.find(p => p.slug === "milk-dip-cup")!,
  products.find(p => p.slug === "harvest-moon-cup")!,
];

export function getProductBySlug(slug: string): Product | undefined {
  return products.find(p => p.slug === slug);
}

export function getRelatedProducts(slug: string, count = 3): Product[] {
  return products.filter(p => p.slug !== slug && p.badge !== "sold-out").slice(0, count);
}
