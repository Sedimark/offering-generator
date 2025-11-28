import json
import uuid
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from config import CONFIG, SEDIMARK_CONTEXT

logger = logging.getLogger(__name__)

class SEDIMARKOfferingGenerator:

    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.schema_data = self._load_schema()

    def _load_schema(self):
        try:
            schema_file = CONFIG.get("schema_file")
            if schema_file and schema_file.exists():
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                logger.info(f"Loaded schema with {len(schema_data.get('@graph', []))} definitions")
                return schema_data
        except Exception as e:
            logger.warning(f"Could not load schema: {e}")
        return None

    def generate_offering(self,
                         prompt: str,
                         use_context: bool = False,
                         context: Dict[str, Any] = None,
                         use_schema: bool = False,
                         **generation_params) -> Dict[str, Any]:

        # Format prompt based on mode
        formatted_prompt = self._format_prompt(prompt, use_context, context, use_schema)

        # Generate
        inputs = self.model_loader.prepare_inputs(formatted_prompt)
        try:
            raw_output = self.model_loader.generate(inputs, **generation_params)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM detected, reducing parameters")
                # Retry with reduced parameters
                fallback_params = {**generation_params, "num_beams": 1, "max_new_tokens": 4096}
                raw_output = self.model_loader.generate(inputs, **fallback_params)
            else:
                raise

        # Extract and validate JSON
        json_text = self._extract_json(raw_output)
        return self._parse_and_validate(json_text, use_schema)

    def _format_prompt(self, prompt: str, use_context: bool, context: Dict, use_schema: bool) -> str:

        if use_context and context:
            return f"""{prompt}



{json.dumps(context, ensure_ascii=False)}



Generate a SEDIMARK-compliant JSON-LD offering with the following structure:
1. @graph array containing all entities
2. @context object at the end
3. Include: Offering, Asset, AssetQuality, AssetProvision, OfferingContract, SelfListing
4. Use proper @id references between entities
5. Follow SEDIMARK ontology strictly

<|json_output|>"""

        elif use_schema and self.schema_data:
            return f"""{prompt}

SCHEMA REFERENCE:
{json.dumps(self.schema_data, indent=2, ensure_ascii=False)}

Instructions: Generate JSON-LD following the exact schema structure provided above.
1. Use the exact class definitions from the schema @graph
2. Follow the @context namespaces exactly as defined
3. Ensure all entities derive from owl:NamedIndividual as specified in the schema
4. Use the exact property names defined in rdfs:properties for each class

<|json_output|>"""

        else:
            return f"""{prompt}
[CONTEXT_EMBEDDED]


Generate a SEDIMARK-compliant JSON-LD offering with the following structure:
1. @graph array containing all entities
2. @context object at the end
3. Include: Offering, Asset, AssetQuality, AssetProvision, OfferingContract, SelfListing
4. Use proper @id references between entities
5. Follow SEDIMARK ontology strictly

<|json_output|>"""

    def _extract_json(self, raw_output: str) -> str:

        # Remove prompt echo
        if "<|json_output|>" in raw_output:
            raw_output = raw_output.split("<|json_output|>")[-1].strip()

        # Remove markdown code blocks
        if "```json" in raw_output:
            match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in raw_output:
            match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Find JSON object/array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = raw_output.find(start_char)
            if start_idx != -1:
                count = 0
                in_string = False
                escape_next = False

                for i in range(start_idx, len(raw_output)):
                    char = raw_output[i]

                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    if not in_string:
                        if char == start_char:
                            count += 1
                        elif char == end_char:
                            count -= 1
                            if count == 0:
                                json_str = raw_output[start_idx:i+1]
                                # Clean common issues
                                json_str = re.sub(r',\s*}', '}', json_str)
                                json_str = re.sub(r',\s*]', ']', json_str)
                                return json_str

        return raw_output.strip()

    def _parse_and_validate(self, json_text: str, use_schema: bool) -> Dict[str, Any]:
        try:
            offering = json.loads(json_text)

            # Ensure proper structure
            if "@context" not in offering:
                offering["@context"] = SEDIMARK_CONTEXT

            # Validate if using schema
            if use_schema and self.schema_data:
                validation_results = self._validate_against_schema(offering)
                offering["_validation"] = validation_results

            return offering

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return self._generate_fallback_offering()

    def _validate_against_schema(self, offering: Dict[str, Any]) -> Dict[str, Any]:
        validation = {
            "timestamp": datetime.now().isoformat(),
            "warnings": [],
            "errors": [],
            "checks": {
                "has_context": "@context" in offering,
                "has_graph": "@graph" in offering,
                "entity_count": len(offering.get("@graph", [])),
            }
        }

        if "@graph" in offering and "@graph" in self.schema_data:
            schema_classes = {item.get("@id") for item in self.schema_data["@graph"] if "@id" in item}
            output_types = []

            for entity in offering["@graph"]:
                if "@type" in entity:
                    entity_types = entity["@type"] if isinstance(entity["@type"], list) else [entity["@type"]]
                    output_types.extend(entity_types)

            used_classes = set(output_types) & schema_classes
            validation["checks"]["schema_coverage"] = len(used_classes) / len(schema_classes) if schema_classes else 0

        return validation

    def _generate_fallback_offering(self) -> Dict[str, Any]:
        offering_id = f"ex:offering_{uuid.uuid4().hex[:8]}"
        asset_id = f"ex:asset_{uuid.uuid4().hex[:8]}"

        return {
            "@graph": [
                {
                    "@id": offering_id,
                    "@type": "sedimark:Offering",
                    "dcterms:title": "Generated IoT Data Offering",
                    "dcterms:description": "Auto-generated offering for IoT sensor data",
                    "sedimark:hasAsset": [{"@id": asset_id}]
                },
                {
                    "@id": asset_id,
                    "@type": "sedimark:Asset",
                    "dcterms:title": "IoT Dataset",
                    "dcterms:description": "Dataset from IoT sensors",
                    "dcterms:issued": {
                        "@value": datetime.now().isoformat() + "Z",
                        "@type": "xsd:dateTime"
                    },
                    "sedimark:offeredBy": {"@id": offering_id}
                }
            ],
            "@context": SEDIMARK_CONTEXT
        }
