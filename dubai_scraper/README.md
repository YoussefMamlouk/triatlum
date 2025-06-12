# District 11 Property Scraper (Optimized)

A high-performance property scraper for District 11 areas in Dubai, optimized for speed and efficiency using direct API calls.

## 🚀 Features

### **Optimized Performance**
- **Direct API Integration**: Uses Bayut's Algolia API and PropertyFinder's __NEXT_DATA__ extraction
- **10x Faster**: Reduced scraping time from hours to minutes
- **Intelligent Duplicate Detection**: Compares `permitNumber` (Bayut) with `rera` field (PropertyFinder)
- **Real-time Processing**: Processes properties as they're fetched

### **Smart Data Collection**
- **Essential Fields Only**: Focuses on core property data
  - Source (Bayut/PropertyFinder)
  - Listing Type (Rent/Sale)  
  - Title
  - Price (AED)
  - Area (sqft)
  - Permit Number
  - Price per sqm (calculated)
- **Cross-Platform Deduplication**: Separate duplicate detection for rent and sale properties
- **Data Quality**: Handles edge cases (studio as bedrooms, price dictionaries, etc.)

### **Target Locations**
- **JVC District 11**: Jumeirah Village Circle
- **MBR City District 11**: Mohammed Bin Rashid City  
- **Sharjah District 11**: Muwaileh Commercial

## 📊 Latest Results

```
🏠 Total unique properties: 2,520
📍 PropertyFinder: 635 properties  
🏢 Bayut: 1,885 properties
🏠 Rental properties: 963
💰 Sale properties: 1,557
📋 Properties with permits: 2,515
🔄 Duplicates removed: 3,071
```

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd triatlum_d11

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Quick Start
```bash
python3 district11_scraper_optimized.py
```

### Output
- **Excel File**: `district11_properties_optimized.xlsx` with multiple sheets:
  - All Properties (main data)
  - PropertyFinder Properties
  - Bayut Properties  
  - Rent Properties
  - Sale Properties
  - With/Without Permits
  - Summary Statistics

## 📁 Project Structure

```
triatlum_d11/
├── district11_scraper_optimized.py    # Main optimized scraper
├── district11_properties_optimized.xlsx # Generated results
├── requirements.txt                    # Dependencies
└── README.md                          # Documentation
```

## 🔧 Technical Details

### **API Integration**
- **Bayut**: Direct Algolia search API with location filters
- **PropertyFinder**: HTML parsing with __NEXT_DATA__ JSON extraction

### **Duplicate Detection Logic**
```python
# Compare permit numbers between sources
bayut_permit = property.get('permitNumber')
pf_rera = property.get('rera')

# Separate processing for rent vs sale
if bayut_permit == pf_rera and listing_type_matches:
    # Keep PropertyFinder version (preferred)
```

### **Performance Optimizations**
- Concurrent API requests with rate limiting
- Memory-efficient data processing
- Smart pagination handling
- Reduced HTTP overhead

## 📈 Data Fields

| Field | Description | Source |
|-------|-------------|---------|
| `source` | Platform name | Bayut/PropertyFinder |
| `listing_type` | Property type | Rent/Sale |
| `title` | Property title | API response |
| `price_aed` | Price in AED | API response |
| `area_sqft` | Area in sq ft | API response |
| `permit_number` | RERA permit | permitNumber/rera field |
| `scraped_at` | Timestamp | Generated |
| `price_per_sqm` | Calculated rate | price_aed / (area_sqft * 0.092903) |

## 🎯 Key Improvements

1. **Speed**: 90%+ reduction in scraping time
2. **Accuracy**: Direct API access eliminates parsing errors  
3. **Reliability**: Robust error handling and retry logic
4. **Cleanliness**: Focused on essential data only
5. **Scalability**: Easy to extend to new locations

## 📋 Requirements

```
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.0.3
openpyxl==3.1.2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. 