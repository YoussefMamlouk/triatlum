import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from datetime import datetime
import os
from collections import defaultdict
import re

class District11OptimizedScraper:
    def __init__(self):
        """Initialize the optimized District 11 scraper using new API-based strategy"""
        self.session = requests.Session()
        self.setup_session()
        self.all_properties = []
        
    def setup_session(self):
        """Setup requests session with realistic headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })

    def fetch_bayut_ads(self, purpose):
        """Fetch ads from Bayut using direct Algolia API calls"""
        url = "https://ll8iz711cs-dsn.algolia.net/1/indexes/*/queries?x-algolia-agent=Algolia%20for%20JavaScript%20(3.35.1)%3B%20Browser%20(lite)&x-algolia-application-id=LL8IZ711CS&x-algolia-api-key=15cb8b0a2d2d435c6613111d860ecfc5"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "fr-FR,fr;q=0.6",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded",
            "Host": "ll8iz711cs-dsn.algolia.net",
            "Origin": "https://www.bayut.com",
            "Referer": "https://www.bayut.com/",
            "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Brave";v="133", "Chromium";v="133"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Gpc": "1",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
        }

        ads = []
        page = 0
        
        print(f"🔍 Fetching Bayut {purpose} properties...")
        
        while True:
            payload = {
                "requests": [
                    {
                        "indexName": "bayut-production-ads-en",
                        "params": f'page={page}&hitsPerPage=24&query=&optionalWords=&facets=[]&maxValuesPerFacet=10&attributesToHighlight=[]&attributesToRetrieve=["type","agency","area","baths","category","additionalCategories","contactName","externalID","sourceID","id","location","objectID","phoneNumber","coverPhoto","photoCount","price","product","productLabel","purpose","geography","permitNumber","referenceNumber","rentFrequency","rooms","slug","slug_l1","slug_l2","slug_l3","title","title_l1","title_l2","title_l3","createdAt","updatedAt","ownerID","isVerified","propertyTour","verification","completionDetails","completionStatus","furnishingStatus","-agency.tier","coverVideo","videoCount","description","description_l1","description_l2","description_l3","descriptionTranslated","descriptionTranslated_l1","descriptionTranslated_l2","descriptionTranslated_l3","floorPlanID","panoramaCount","hasMatchingFloorPlans","state","photoIDs","reactivatedAt","hidePrice","extraFields","projectNumber","locationPurposeTier","hasRedirectionLink","ownerAgent","hasEmail","plotArea","offplanDetails","paymentPlans","paymentPlanSummaries","project","availabilityStatus","userExternalID","units","unitCategories","downPayment","clips","contactMethodAvailability","agentAdStoriesCount"]&filters=purpose:"{purpose}"%20AND%20(location.slug:"/dubai/jumeirah-village-circle-jvc/jvc-district-11"%20OR%20location.slug:"/dubai/mohammed-bin-rashid-city/district-11"%20OR%20location.slug:"/sharjah/muwaileh-commercial/district-11")%20AND%20category.slug:"residential"'
                    }
                ]
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                hits = data['results'][0]['hits']
                if not hits:
                    break  # No more pages
                
                ads.extend(hits)
                print(f"    Fetched page {page} with {len(hits)} ads")
                page += 1
                
                # Small delay to be respectful to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error fetching page {page}: {e}")
                break

        print(f"    ✅ Total {purpose} ads fetched: {len(ads)}")
        return ads

    def fetch_propertyfinder_listings(self, category_id, category_name):
        """Fetch PropertyFinder listings using __NEXT_DATA__ extraction"""
        base_url = "https://www.propertyfinder.ae/en/search"
        params = {
            "l": "1320-1037-15083",
            "c": str(category_id),
            "fu": "0",
            "ob": "mr",
            "page": 1
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        all_listings = []
        seen_ids = set()
        page = 1
        
        print(f"🔍 Fetching PropertyFinder {category_name} properties...")

        while True:
            print(f"    Fetching page {page}...")
            params["page"] = page
            
            try:
                res = self.session.get(base_url, headers=headers, params=params)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                script_tag = soup.find("script", {"id": "__NEXT_DATA__"})

                if not script_tag or not script_tag.string:
                    print(f"    Page {page}: __NEXT_DATA__ not found or empty.")
                    break

                data = json.loads(script_tag.string)
                page_props = data.get("props", {}).get("pageProps", {})
                search_result = page_props.get("searchResult")

                if not search_result or "listings" not in search_result:
                    print(f"    Page {page}: 'listings' not found.")
                    break

                listings = search_result["listings"]
                new_listings = []

                for l in listings:
                    prop = l.get("property")
                    if prop and prop.get("id") and prop["id"] not in seen_ids:
                        seen_ids.add(prop["id"])
                        new_listings.append(l)

                if not new_listings:
                    print(f"    No new listings on page {page}. Likely end.")
                    break

                all_listings.extend(new_listings)
                page += 1
                time.sleep(1)  # polite delay
                
            except Exception as e:
                print(f"    ❌ Error parsing page {page}: {e}")
                break

        print(f"    ✅ Total {category_name} listings fetched: {len(all_listings)}")
        return all_listings

    def process_bayut_property(self, hit, listing_type):
        """Process a single Bayut property from API response"""
        try:
            # Extract basic info
            title = hit.get('title', '')
            price_aed = hit.get('price', 0)
            area_sqft = hit.get('area', 0)
            
            # Handle permit number safely - it might be None
            permit_raw = hit.get('permitNumber', '')
            permit_number = permit_raw.strip().upper() if permit_raw else ''
            
            return {
                'source': 'Bayut',
                'listing_type': listing_type,
                'title': title,
                'price_aed': float(price_aed) if price_aed else 0,
                'area_sqft': float(area_sqft) if area_sqft else 0,
                'permit_number': permit_number,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    ⚠️  Error processing Bayut property: {e}")
            return None

    def process_propertyfinder_property(self, listing, listing_type):
        """Process a single PropertyFinder property from API response"""
        try:
            prop = listing.get('property', {})
            
            # Extract basic info
            title = prop.get('title', '')
            
            # Handle price - it's a dictionary with 'value' key
            price_data = prop.get('price', {})
            if isinstance(price_data, dict):
                price_aed = price_data.get('value', 0)
            else:
                price_aed = price_data or 0
            
            # Handle area/size - check multiple possible field names
            area_sqft = 0
            for size_field in ['area', 'size', 'area_sqft']:
                if size_field in prop:
                    size_data = prop[size_field]
                    if isinstance(size_data, dict):
                        area_sqft = size_data.get('value', 0) or size_data.get('area', 0)
                    else:
                        area_sqft = size_data or 0
                    if area_sqft:
                        break
            
            # Extract RERA permit number (this is the key field for PropertyFinder)
            rera_raw = prop.get('rera', '')
            rera_number = rera_raw.strip().upper() if rera_raw else ''
            
            return {
                'source': 'PropertyFinder',
                'listing_type': listing_type,
                'title': title,
                'price_aed': float(price_aed) if price_aed else 0,
                'area_sqft': float(area_sqft) if area_sqft else 0,
                'permit_number': rera_number,  # Use rera as permit_number for consistency
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    ⚠️  Error processing PropertyFinder property: {e}")
            return None

    def scrape_all_properties(self):
        """Scrape only sale properties from both sources"""
        all_properties = []
        
        print("🚀 Starting optimized scraping for BUY properties only...")
        print("=" * 60)
        
        # Fetch Bayut properties
        print("\n📍 BAYUT PROPERTIES")
        print("-" * 30)
        
        # Fetch only sale properties
        bayut_sale_ads = self.fetch_bayut_ads("for-sale")
        for hit in bayut_sale_ads:
            prop = self.process_bayut_property(hit, "Sale")
            if prop:
                all_properties.append(prop)
        
        # Fetch PropertyFinder properties
        print("\n🏢 PROPERTYFINDER PROPERTIES")
        print("-" * 30)
        
        # Fetch only sale properties (category 1)
        pf_sale_listings = self.fetch_propertyfinder_listings(1, "sale")
        for listing in pf_sale_listings:
            prop = self.process_propertyfinder_property(listing, "Sale")
            if prop:
                all_properties.append(prop)
        
        print(f"\n✅ Total sale properties scraped: {len(all_properties)}")
        return all_properties

    def detect_duplicates(self, properties):
        """Detect duplicates by comparing permit numbers for sale properties"""
        duplicates = []
        
        # Create duplicates log file
        log_filename = f'duplicates_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(log_filename, 'w', encoding='utf-8') as log_file:
            log_file.write("DISTRICT 11 SALE PROPERTY DUPLICATES LOG\n")
            log_file.write("=" * 50 + "\n")
            log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
            print(f"\n🔍 Checking Sale duplicates...")
            log_file.write(f"SALE PROPERTIES DUPLICATES\n")
            log_file.write("-" * 30 + "\n")
            
            # Group by permit number (excluding empty ones)
            permit_groups = defaultdict(list)
            no_permit_props = []
            
            for prop in properties:
                permit = prop.get('permit_number', '').strip()
                if permit:
                    permit_groups[permit].append(prop)
                else:
                    no_permit_props.append(prop)
            
            # Find duplicates
            duplicate_groups = {permit: props_list for permit, props_list in permit_groups.items() 
                              if len(props_list) > 1}
            
            if duplicate_groups:
                print(f"    Found {len(duplicate_groups)} duplicate permit numbers in Sale:")
                log_file.write(f"Found {len(duplicate_groups)} duplicate permit groups:\n\n")
                
                for permit, props_list in duplicate_groups.items():
                    print(f"      🔗 Permit {permit}: {len(props_list)} properties")
                    log_file.write(f"PERMIT: {permit} ({len(props_list)} properties)\n")
                    
                    for i, prop in enumerate(props_list, 1):
                        print(f"        - {prop['source']}: {prop['title'][:50]}...")
                        log_file.write(f"  {i}. {prop['source']}: {prop['title']}\n")
                        log_file.write(f"     Price: {prop['price_aed']:,} AED\n")
                        log_file.write(f"     Area: {prop['area_sqft']} sqft\n")
                        log_file.write(f"     Scraped: {prop['scraped_at']}\n")
                    
                    log_file.write("\n")
                    duplicates.extend(props_list)
            else:
                print(f"    ✅ No duplicates found in Sale")
                log_file.write(f"✅ No duplicates found in Sale\n")
            
            if no_permit_props:
                print(f"    ⚠️  {len(no_permit_props)} Sale properties without permit numbers")
                log_file.write(f"⚠️  {len(no_permit_props)} properties without permit numbers\n")
            
            log_file.write("\n")
        
        print(f"\n📝 Duplicates logged to: {log_filename}")
        return duplicates

    def remove_duplicates(self, properties):
        """Remove duplicates, keeping PropertyFinder properties when duplicates exist"""
        print("\n🔄 Removing duplicates...")
        
        # Group by permit number
        permit_groups = defaultdict(list)
        no_permit_props = []
        
        for prop in properties:
            permit = prop.get('permit_number', '').strip()
            if permit:
                permit_groups[permit].append(prop)
            else:
                no_permit_props.append(prop)
        
        unique_properties = []
        
        # For each permit group, keep PropertyFinder if available, otherwise keep the first one
        for permit, props_list in permit_groups.items():
            if len(props_list) == 1:
                unique_properties.append(props_list[0])
            else:
                # Prefer PropertyFinder over Bayut
                pf_props = [p for p in props_list if p['source'] == 'PropertyFinder']
                if pf_props:
                    unique_properties.append(pf_props[0])  # Keep first PropertyFinder property
                else:
                    unique_properties.append(props_list[0])  # Keep first Bayut property
        
        # Add properties without permit numbers (keep all for now)
        unique_properties.extend(no_permit_props)
        
        removed_count = len(properties) - len(unique_properties)
        print(f"    Removed {removed_count} duplicate properties")
        print(f"    Kept {len(unique_properties)} unique properties")
        
        return unique_properties

    def save_to_excel(self, properties, filename='district11_properties_optimized.xlsx'):
        """Save sale properties to Excel with detailed analysis"""
        if not properties:
            print("❌ No properties to save")
            return
        
        df = pd.DataFrame(properties)
        
        # Calculate price per sqm for sale properties only
        mask = (df['price_aed'] > 0) & (df['area_sqft'] > 0)
        
        # Initialize price per sqm column
        df['price_per_sqm'] = 0
        
        # Calculate price per sqm for sale properties  
        df.loc[mask, 'price_per_sqm'] = (df.loc[mask, 'price_aed'] / (df.loc[mask, 'area_sqft'] * 0.092903)).round(2)
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # All properties
            df.to_excel(writer, sheet_name='All Properties', index=False)
            
            # Separate sheets by source
            for source in df['source'].unique():
                df[df['source'] == source].to_excel(writer, sheet_name=f'{source} Properties', index=False)
            
            # Properties with and without permits
            permit_mask = df['permit_number'].notna() & (df['permit_number'] != '')
            if permit_mask.any():
                df[permit_mask].to_excel(writer, sheet_name='With Permits', index=False)
            if (~permit_mask).any():
                df[~permit_mask].to_excel(writer, sheet_name='Without Permits', index=False)
            
            # Summary statistics for sale properties
            avg_sale_price_per_sqm = df[df['price_per_sqm'] > 0]['price_per_sqm'].mean()
            
            summary_data = {
                'Metric': [
                    'Total Properties',
                    'PropertyFinder Properties', 
                    'Bayut Properties',
                    'Properties with Permits',
                    'Average Sale Price (AED)',
                    'Average Area (sqft)',
                    'Average Sale Price per sqm (AED)'
                ],
                'Value': [
                    len(df),
                    len(df[df['source'] == 'PropertyFinder']),
                    len(df[df['source'] == 'Bayut']),
                    permit_mask.sum(),
                    f"{df[df['price_aed'] > 0]['price_aed'].mean():,.0f}" if len(df[df['price_aed'] > 0]) > 0 else 'N/A',
                    f"{df[df['area_sqft'] > 0]['area_sqft'].mean():,.0f}" if len(df[df['area_sqft'] > 0]) > 0 else 'N/A',
                    f"{avg_sale_price_per_sqm:,.0f}" if not pd.isna(avg_sale_price_per_sqm) else 'N/A'
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"💾 Saved {len(properties)} properties to {filename}")
        print(f"📊 Added price per sqm analysis for sale properties")
        return filename

    def run_optimized_scrape(self):
        """Run the complete optimized scraping process for buy properties only"""
        print("🚀 District 11 BUY Property Scraper (Sale Properties Only)")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Scrape only sale properties
        all_properties = self.scrape_all_properties()
        
        # Analyze and detect duplicates
        print("\n🔍 DUPLICATE ANALYSIS")
        print("-" * 30)
        duplicates = self.detect_duplicates(all_properties)
        
        # Remove duplicates
        unique_properties = self.remove_duplicates(all_properties)
        
        # Save to Excel
        print("\n💾 SAVING RESULTS")
        print("-" * 30)
        filename = self.save_to_excel(unique_properties)
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        
        final_pf = len([p for p in unique_properties if p['source'] == 'PropertyFinder'])
        final_bayut = len([p for p in unique_properties if p['source'] == 'Bayut'])
        with_permits = len([p for p in unique_properties if p.get('permit_number', '').strip()])
        
        print(f"🏠 Total unique sale properties: {len(unique_properties)}")
        print(f"📍 PropertyFinder: {final_pf} properties")
        print(f"🏢 Bayut: {final_bayut} properties")
        print(f"📋 Properties with permits: {with_permits}")
        print(f"💾 Saved to: {filename}")
        print(f"⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return unique_properties

if __name__ == "__main__":
    scraper = District11OptimizedScraper()
    properties = scraper.run_optimized_scrape() 