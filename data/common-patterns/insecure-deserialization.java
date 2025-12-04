// Common Insecure Deserialization Pattern - For Cache Warming
// This pattern should be detected as a security vulnerability

import java.io.*;

public class DeserializationExample {
    
    // VULNERABLE: Deserializing untrusted data
    public Object deserializeUnsafe(byte[] data) throws Exception {
        // DANGEROUS: Direct deserialization of untrusted input
        ByteArrayInputStream bis = new ByteArrayInputStream(data);
        ObjectInputStream ois = new ObjectInputStream(bis);
        return ois.readObject();  // Remote Code Execution possible
    }
    
    // VULNERABLE: Reading serialized object from file
    public Object readFromFileUnsafe(String filename) throws Exception {
        FileInputStream fis = new FileInputStream(filename);
        ObjectInputStream ois = new ObjectInputStream(fis);
        return ois.readObject();  // No validation of class type
    }
    
    // SAFE: Using whitelist-based deserialization
    public Object deserializeSafe(byte[] data, Class<?>[] allowedClasses) throws Exception {
        ByteArrayInputStream bis = new ByteArrayInputStream(data);
        
        // Use ObjectInputFilter (Java 9+)
        ObjectInputStream ois = new ObjectInputStream(bis) {
            @Override
            protected Class<?> resolveClass(ObjectStreamClass desc) 
                    throws IOException, ClassNotFoundException {
                String className = desc.getName();
                
                // Whitelist check
                for (Class<?> allowed : allowedClasses) {
                    if (allowed.getName().equals(className)) {
                        return super.resolveClass(desc);
                    }
                }
                
                throw new InvalidClassException("Unauthorized class: " + className);
            }
        };
        
        return ois.readObject();
    }
    
    // SAFE: Using JSON instead of Java serialization
    public User deserializeJson(String json) {
        // Use Jackson or Gson with type information disabled
        ObjectMapper mapper = new ObjectMapper();
        mapper.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
        mapper.activateDefaultTyping(
            mapper.getPolymorphicTypeValidator(),
            ObjectMapper.DefaultTyping.NON_FINAL
        );
        return mapper.readValue(json, User.class);
    }
}

class User implements Serializable {
    private String name;
    private String email;
    
    // Getters and setters
}
