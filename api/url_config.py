#!/usr/bin/env python3
"""
URL Configuration File for JSON-LD Generator

This file contains all URL configurations except standard namespaces.
Modify the URLs here to customize entity URLs and base URIs for different projects.
"""

# Base URI configurations
BASE_URI_CONFIGS = {
    "sedimark": "https://sedimark.surrey.ac.uk/ecosystem",
    "surrey": "https://surrey.ac.uk/data",
    "example": "http://example.org",
    "default": "https://sedimark.surrey.ac.uk/ecosystem"
}

# Entity URL patterns
# These will be combined with base URI to create full entity URLs
ENTITY_URL_PATTERNS = {
    "SelfListing": "self-listing",
    "Participant": "participant", 
    "DataAsset": "data-asset",
    "AssetQuality": "asset-quality",
    "Location": "location",
    "PeriodOfTime": "period-of-time",
    "OfferingContract": "offering-contract",
    "Offering": "offering"
}

# Custom entity URLs (override patterns if needed)
# Format: entity_name: full_url
CUSTOM_ENTITY_URLS = {
    # Example custom URLs - uncomment and modify as needed
    # "SelfListing": "https://custom.org/catalog/main-listing",
    # "DataAsset": "https://custom.org/datasets/primary-asset",
}

# Ontology URL patterns
ONTOLOGY_PATTERNS = {
    "sedi_suffix": "/ontology/sedi#",
    "vocab_suffix": "/ontology#"
}

# Additional namespace configurations (non-standard)
ADDITIONAL_NAMESPACES = {
    # Add custom namespaces here if needed
    # "custom": "https://custom.org/vocab#",
    # "project": "https://project.org/terms#"
}

# Domain-specific URL configurations
DOMAIN_CONFIGS = {
    "health": {
        "base_uri": "https://health.sedimark.ac.uk",
        "entity_patterns": {
            "DataAsset": "health-dataset",
            "Location": "health-facility"
        }
    },
    "smart_city": {
        "base_uri": "https://smartcity.sedimark.ac.uk", 
        "entity_patterns": {
            "DataAsset": "city-dataset",
            "Location": "city-location"
        }
    },
    "iot": {
        "base_uri": "https://iot.sedimark.ac.uk",
        "entity_patterns": {
            "DataAsset": "sensor-dataset", 
            "Location": "sensor-location"
        }
    }
}


class URLConfigManager:
    """Manages URL configurations for JSON-LD generation"""
    
    def __init__(self, base_uri: str = None, domain: str = None):
        """
        Initialize URL configuration manager
        
        Args:
            base_uri: Custom base URI to use
            domain: Domain configuration to use (health, smart_city, iot, etc.)
        """
        self.base_uri = base_uri or BASE_URI_CONFIGS["default"]
        self.domain = domain
        
        # Load domain-specific configurations if specified
        if domain and domain in DOMAIN_CONFIGS:
            domain_config = DOMAIN_CONFIGS[domain]
            if not base_uri:  # Only use domain base_uri if not explicitly provided
                self.base_uri = domain_config["base_uri"]
            self.entity_patterns = {**ENTITY_URL_PATTERNS, **domain_config.get("entity_patterns", {})}
        else:
            self.entity_patterns = ENTITY_URL_PATTERNS.copy()
    
    def get_base_uri(self) -> str:
        """Get the configured base URI"""
        return self.base_uri
    
    def get_entity_url(self, entity_name: str) -> str:
        """
        Get full URL for an entity
        
        Args:
            entity_name: Name of the entity (e.g., 'SelfListing')
            
        Returns:
            Full URL for the entity
        """
        # Check for custom URLs first
        if entity_name in CUSTOM_ENTITY_URLS:
            return CUSTOM_ENTITY_URLS[entity_name]
        
        # Use pattern-based URL generation
        pattern = self.entity_patterns.get(entity_name, entity_name.lower())
        return f"{self.base_uri}/{pattern}"
    
    def get_all_entity_urls(self) -> dict:
        """Get all entity URLs as a dictionary"""
        return {entity: self.get_entity_url(entity) for entity in ENTITY_URL_PATTERNS.keys()}
    
    def get_sedi_namespace(self) -> str:
        """Get SEDI namespace URL"""
        return f"{self.base_uri}{ONTOLOGY_PATTERNS['sedi_suffix']}"
    
    def get_vocab_namespace(self) -> str:
        """Get @vocab namespace URL"""
        return f"{self.base_uri}{ONTOLOGY_PATTERNS['vocab_suffix']}"
    
    def get_standard_namespaces(self) -> dict:
        """Get standard namespaces dictionary"""
        standard = {
            "sedi": self.get_sedi_namespace(),
            "dct": "http://purl.org/dc/terms/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "dcat": "http://www.w3.org/ns/dcat#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "@vocab": self.get_vocab_namespace()
        }
        
        # Add additional namespaces if configured
        standard.update(ADDITIONAL_NAMESPACES)
        
        return standard
    
    def get_config_summary(self) -> dict:
        """Get a summary of current configuration"""
        return {
            "base_uri": self.base_uri,
            "domain": self.domain,
            "entity_urls": self.get_all_entity_urls(),
            "namespaces": self.get_standard_namespaces()
        }


# Convenience functions
def get_url_config(base_uri: str = None, domain: str = None) -> URLConfigManager:
    """Get a URL configuration manager instance"""
    return URLConfigManager(base_uri, domain)

def get_entity_urls(base_uri: str = None, domain: str = None) -> dict:
    """Get all entity URLs for a given configuration"""
    config = URLConfigManager(base_uri, domain)
    return config.get_all_entity_urls()

def get_namespaces(base_uri: str = None, domain: str = None) -> dict:
    """Get all namespaces for a given configuration"""
    config = URLConfigManager(base_uri, domain)
    return config.get_standard_namespaces()


# Example usage and testing
if __name__ == "__main__":
    print("URL Configuration Examples:")
    print("=" * 50)
    
    # Default configuration
    default_config = URLConfigManager()
    print(f"Default Base URI: {default_config.get_base_uri()}")
    print(f"Default SelfListing URL: {default_config.get_entity_url('SelfListing')}")
    
    # Custom base URI
    custom_config = URLConfigManager(base_uri="https://myorg.com/data")
    print(f"\nCustom Base URI: {custom_config.get_base_uri()}")
    print(f"Custom SelfListing URL: {custom_config.get_entity_url('SelfListing')}")
    
    # Domain-specific configuration
    health_config = URLConfigManager(domain="health")
    print(f"\nHealth Domain Base URI: {health_config.get_base_uri()}")
    print(f"Health DataAsset URL: {health_config.get_entity_url('DataAsset')}")
    
    # Show all entity URLs
    print(f"\nAll Entity URLs (default):")
    for entity, url in default_config.get_all_entity_urls().items():
        print(f"  {entity}: {url}")
    
    # Show all namespaces
    print(f"\nAll Namespaces (default):")
    for prefix, url in default_config.get_standard_namespaces().items():
        print(f"  {prefix}: {url}")