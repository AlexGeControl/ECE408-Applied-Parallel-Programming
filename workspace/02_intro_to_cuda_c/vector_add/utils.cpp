#include "utils.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/uuid/detail/md5.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace pt = boost::property_tree;

void readInputJSON(Vector& x, Vector& y, const std::string& inputJSON) {
    // Read input JSON file
    pt::ptree root;
    pt::read_json(inputJSON, root);

    // Extract x and y arrays into vectors
    for (const auto& item : root.get_child("x")) {
        x.push_back(item.second.get_value<double>());
    }
    
    for (const auto& item : root.get_child("y")) {
        y.push_back(item.second.get_value<double>());
    }

    std::cout << "Loaded vectors with size: " << x.size() << std::endl;
}

void writeOutputJSON(const Vector& result, const std::string& outputJSON) {
    // Write result to output JSON file
    pt::ptree root;
    pt::ptree result_array;
    
    for (const auto& value : result) {
        pt::ptree array_element;
        array_element.put("", value);
        result_array.push_back(std::make_pair("", array_element));
    }
    
    root.add_child("result", result_array);
    pt::write_json(outputJSON, root);
    
    std::cout << "Results written to: " << outputJSON << std::endl;
}

std::string computeMD5(const Vector& result) {
    boost::uuids::detail::md5 hash;
    boost::uuids::detail::md5::digest_type digest;
    
    // Hash the raw bytes of the vector data
    hash.process_bytes(result.data(), result.size() * sizeof(ElementType));
    hash.get_digest(digest);
    
    // Convert the 127-bit digest to 32-character hex string
    std::ostringstream oss;
    for (int i = 0; i < 4; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(8) << digest[i];
    }
    
    return oss.str();
}
