import networkx as nx

def calculate_xmax_ymax(bbox):
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    return xmax, ymax

def calculate_spatial_relations(bbox1, bbox2):
    xmin1, ymin1, width1, height1 = bbox1
    xmin2, ymin2, width2, height2 = bbox2
    
    xmax1, ymax1 = calculate_xmax_ymax(bbox1)
    xmax2, ymax2 = calculate_xmax_ymax(bbox2)
    
    relations = []
    
    if xmin1 < xmax2 and xmax1 > xmin2 and ymin1 < ymax2 and ymax1 > ymin2:
        relations.append("overlaps")

    if xmax1 < xmin2:
        relations.append("left_of")

    if xmin1 > xmax2:
        relations.append("right_of")

    if ymax1 < ymin2:
        relations.append("above")

    if ymin1 > ymax2:
        relations.append("below")
    
    return relations


def relation_to_text(source_id, source_label, relation, target_id, target_label):
    if relation == "overlaps":
        return f"Object {source_id} ({source_label}) overlaps with Object {target_id} ({target_label})."
    elif relation == "left_of":
        return f"Object {source_id} ({source_label}) is to the left of Object {target_id} ({target_label})."
    elif relation == "right_of":
        return f"Object {source_id} ({source_label}) is to the right of Object {target_id} ({target_label})."
    elif relation == "above":
        return f"Object {source_id} ({source_label}) is above Object {target_id} ({target_label})."
    elif relation == "below":
        return f"Object {source_id} ({source_label}) is below Object {target_id} ({target_label})."
    elif relation == "same_object_type":
        return f"Object {source_id} ({source_label}) is of the same type as Object {target_id} ({target_label})."
    else:
        return f"Object {source_id} ({source_label}) is related to Object {target_id} ({target_label})."

def generate_scene_graph_description(objects, location_des, relation_des, number_des):

    scene_graph = nx.DiGraph()
    object_count = {}

    for obj in objects:
        scene_graph.add_node(obj['id'], label=obj['label'], bbox=obj['bbox'])

        label = obj['label']
        if label in object_count:
            object_count[label] += 1
        else:
            object_count[label] = 1

    for node1, data1 in scene_graph.nodes(data=True):
        for node2, data2 in scene_graph.nodes(data=True):
            if node1 < node2:
                bbox1 = data1['bbox']
                bbox2 = data2['bbox']
                relations = calculate_spatial_relations(bbox1, bbox2)
                
                for relation in relations:
                    scene_graph.add_edge(node1, node2, relation=relation)

    descriptions = []

    if location_des:
        for node, data in scene_graph.nodes(data=True):
            label = data.get('label', 'unknown object')
            bbox = data.get('bbox', [])
            description = f"Object {node} is a {label} located at coordinates [{bbox[0]}, {bbox[1]}] with dimensions {bbox[2]}x{bbox[3]}."
            descriptions.append(description)
    
    if relation_des:
        for source, target, data in scene_graph.edges(data=True):
            relation = data.get('relation', 'related to')
            source_label = scene_graph.nodes[source]['label']
            target_label = scene_graph.nodes[target]['label']
            description = relation_to_text(source, source_label, relation, target, target_label)
            descriptions.append(description)

    if number_des:
        count_description = "Object counting:\n"
        for label, count in object_count.items():
            count_description += f"- {label}: {count}\n"
        descriptions.append(count_description)
    
    return "\n".join(descriptions)